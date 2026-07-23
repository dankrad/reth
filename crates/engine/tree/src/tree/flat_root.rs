//! FLATMPT experiment: state roots from the flat MPT instead of the MDBX trie
//! tables (`RETH_FLATMPT_ROOT=1`, flat file from `FLATMPT`).
//!
//! Replicates the tempo-node follower semantics on the Ethereum side:
//! - Live payload validation computes the block's root via an optimistic
//!   [`FlatShadowLite::root_for`] (apply, memoize, keep the inverse diff);
//!   candidates that lose fork-choice unwind through the inverse diffs.
//! - Pipeline/backfill ranges (which bypass payload validation) are fed by
//!   the shadow ExEx through [`FlatShadowLite::apply_committed`], which skips
//!   blocks the live path already applied and unwinds mismatched candidates.
//!
//! The hook returns EMPTY trie updates: the (frozen, decorative) MDBX trie
//! tables are never written under the flag.

use alloy_primitives::{keccak256, B256, U256};
use mpt_flat_poc::{FlatMpt, Key, StateOp};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Reorg window: candidate blocks whose inverse diffs we retain.
const INVERSE_WINDOW: u64 = 128;
/// Flat-file persist cadence (blocks).
const PERSIST_EVERY: u64 = 100;

/// `RETH_FLATMPT_ROOT=1` — flat MPT replaces the trie tables end to end.
pub fn flatmpt_root_mode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RETH_FLATMPT_ROOT").as_deref() == Ok("1"))
}

/// The process-wide flat shadow (validation hook + ExEx feeder share it).
pub fn shadow() -> &'static Mutex<FlatShadowLite> {
    static S: OnceLock<Mutex<FlatShadowLite>> = OnceLock::new();
    S.get_or_init(|| {
        let s = Mutex::new(FlatShadowLite::open().expect("FLATMPT flat shadow failed to open"));
        spawn_bg_gc();
        s
    })
}

/// Background GC (tempo's collect/install split, eth-sized): collect victim
/// regions against a pinned snapshot without holding the shadow lock, then
/// briefly re-lock to verify+install. Mainnet's 12s cadence leaves the lock
/// idle almost always; try_lock keeps gc strictly off the apply path.
/// `FLATMPT_BG_GC=0` disables; `FLATMPT_GC_CHUNK` regions/cycle (default 512).
fn spawn_bg_gc() {
    if std::env::var("FLATMPT_BG_GC").as_deref() == Ok("0") {
        return;
    }
    let chunk: usize = std::env::var("FLATMPT_GC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    std::thread::Builder::new()
        .name("flatmpt-gc".into())
        .spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let snap = match shadow().try_lock() {
                Ok(g) => g.db.snapshot(),
                Err(_) => continue,
            };
            let t_collect = std::time::Instant::now();
            let batch = match snap.gc_collect(chunk) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(target: "flatmpt", err = %format!("{e:#}"), "bg gc collect failed");
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    continue;
                }
            };
            if batch.is_empty() {
                std::thread::sleep(std::time::Duration::from_secs(30));
                continue;
            }
            let collect_ms = t_collect.elapsed().as_millis() as u64;
            let (items, regions) = (batch.len(), batch.regions());
            // Install in the next idle moment; the batch's snapshot pin keeps
            // it sound while we wait. Give up after 30s (dropped batch =
            // wasted relocation writes, never wrong data).
            let mut batch = Some(batch);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while let Some(b) = batch.take() {
                match shadow().try_lock() {
                    Ok(mut g) => match g.db.gc_install(b) {
                        Ok((installed, discarded)) => {
                            g.db.prefetch_clear(); // release staged regions
                            tracing::debug!(
                                target: "flatmpt",
                                regions, items, installed, discarded, collect_ms,
                                "bg gc cycle"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(target: "flatmpt", err = %format!("{e:#}"), "bg gc install failed")
                        }
                    },
                    Err(_) => {
                        if std::time::Instant::now() < deadline {
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            batch = Some(b);
                        } else {
                            tracing::debug!(target: "flatmpt", regions, items, "bg gc batch dropped (lock busy)");
                        }
                    }
                }
            }
        })
        .expect("spawn flatmpt-gc");
}

pub struct FlatShadowLite {
    db: FlatMpt,
    path: String,
    /// Highest applied block.
    height: u64,
    head_hash: B256,
    /// Applied blocks by number: (hash, inverse ops) — the unwind window.
    applied: BTreeMap<u64, (B256, Vec<(Key, StateOp)>)>,
    /// Roots of validated candidates by block hash (revalidation memo).
    memo: std::collections::HashMap<B256, B256>,
    blocks_since_persist: u64,
}

impl FlatShadowLite {
    fn open() -> anyhow::Result<Self> {
        let path = std::env::var("FLATMPT")
            .map_err(|_| anyhow::anyhow!("RETH_FLATMPT_ROOT=1 needs FLATMPT=<flat path>"))?;
        let db = FlatMpt::open(&path)?;
        let height_file = std::fs::read_to_string(format!("{path}.height"))
            .map_err(|_| anyhow::anyhow!("missing {path}.height (\"<number> <hash>\")"))?;
        let mut it = height_file.split_whitespace();
        let height: u64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("bad {path}.height"))?;
        let head_hash: B256 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("{path}.height needs \"<number> <hash>\""))?;
        // Third field (new format): the flat root at checkpoint time. A
        // mismatch means the file and the checkpoint tore apart (kill -9
        // between persist and height write, or a divergence persisted) —
        // replaying from the stale height onto a future state is NOT
        // idempotent, so fail loudly instead of corrupting.
        if let Some(root) = it.next() {
            let want: B256 = root
                .parse()
                .map_err(|_| anyhow::anyhow!("bad root field in {path}.height"))?;
            let have = B256::from(db.root());
            if want != have && std::env::var("FLATMPT_HEAL").as_deref() != Ok("1") {
                anyhow::bail!(
                    "flat file root {have} != checkpoint root {want} at block {height} — torn checkpoint;                      set FLATMPT_HEAL=1 to converge by replaying forward"
                );
            }
        }
        tracing::info!(target: "flatmpt", height, %head_hash, "flat shadow (engine) opened");
        Ok(Self {
            db,
            path,
            height,
            head_hash,
            applied: BTreeMap::new(),
            memo: Default::default(),
            blocks_since_persist: 0,
        })
    }

    fn persist(&mut self) -> anyhow::Result<()> {
        self.db.persist()?;
        std::fs::write(
            format!("{}.height", self.path),
            format!("{} {} {}\n", self.height, self.head_hash, B256::from(self.db.root())),
        )?;
        Ok(())
    }

    /// Unwind applied blocks above `to` (exclusive) through inverse diffs.
    fn unwind_to(&mut self, to: u64, to_hash: B256) -> anyhow::Result<()> {
        while self.height > to {
            let n = self.height;
            let (hash, inverse) = self
                .applied
                .remove(&n)
                .ok_or_else(|| anyhow::anyhow!("no inverse retained for block {n} (reorg past window)"))?;
            let (_, _) = self.db.apply_block(inverse)?;
            self.memo.remove(&hash);
            self.height = n - 1;
            self.head_hash = self
                .applied
                .get(&self.height)
                .map(|(h, _)| *h)
                .unwrap_or(to_hash);
            tracing::info!(target: "flatmpt", block = n, "flat shadow unwound");
        }
        if self.height == to && self.head_hash != to_hash {
            anyhow::bail!(
                "unwound to block {to} but head is {} not {to_hash}",
                self.head_hash
            );
        }
        Ok(())
    }

    /// Root after applying `ops` for candidate `(number, hash)` on parent
    /// `(number-1, parent_hash)`. Optimistic: the candidate stays applied;
    /// a later sibling candidate unwinds it first. Memoized per block hash.
    pub fn root_for(
        &mut self,
        number: u64,
        hash: B256,
        parent_hash: B256,
        ops: Vec<(Key, StateOp)>,
    ) -> anyhow::Result<B256> {
        if let Some(root) = self.memo.get(&hash) {
            return Ok(*root);
        }
        if number <= self.height {
            // Sibling/deeper fork: unwind to the fork point.
            self.unwind_to(number - 1, parent_hash)?;
        }
        if number != self.height + 1 || parent_hash != self.head_hash {
            anyhow::bail!(
                "flat shadow at {} ({}), can't serve candidate {number} on parent {parent_hash}",
                self.height,
                self.head_hash
            );
        }
        let (root, inverse) = self.db.apply_block(ops)?;
        self.height = number;
        self.head_hash = hash;
        self.applied.insert(number, (hash, inverse));
        self.applied.retain(|n, _| *n + INVERSE_WINDOW > number);
        let root = B256::from(root);
        self.memo.insert(hash, root);
        if self.memo.len() > 4096 {
            self.memo.clear(); // cheap bound; entries re-derive via applied state
        }
        self.blocks_since_persist += 1;
        if self.blocks_since_persist >= PERSIST_EVERY {
            self.persist()?;
            self.blocks_since_persist = 0;
        }
        Ok(root)
    }

    /// Apply a committed range delivered by the ExEx (pipeline/backfill or
    /// canonical commits). Blocks already applied by the live path (same
    /// number AND hash) are skipped; a mismatched hash at an applied height
    /// unwinds the stale candidates first. Returns the last applied root.
    pub fn apply_committed(
        &mut self,
        first: u64,
        tip: u64,
        tip_hash: B256,
        parent_hash: B256,
        ops: Vec<(Key, StateOp)>,
    ) -> anyhow::Result<Option<B256>> {
        if tip <= self.height {
            // Fully known — verify lineage where we can.
            if let Some((h, _)) = self.applied.get(&tip) {
                if *h != tip_hash {
                    anyhow::bail!("committed block {tip} hash {tip_hash} != applied {h}");
                }
            }
            return Ok(None);
        }
        if first <= self.height {
            // Straddling range: the aggregated ops cover blocks we already
            // applied. The caller (ExEx) splits the outcome so ops here only
            // cover height+1..=tip.
        } else if first != self.height + 1 {
            anyhow::bail!("gap: committed {first}..={tip}, shadow at {}", self.height);
        }
        let applied_from = self.height + 1;
        let (root, inverse) = self.db.apply_block(ops)?;
        self.height = tip;
        self.head_hash = tip_hash;
        let _ = parent_hash;
        self.applied.insert(tip, (tip_hash, inverse));
        self.applied.retain(|n, _| *n + INVERSE_WINDOW > tip);
        self.memo.insert(tip_hash, B256::from(root));
        // Count BLOCKS, not notifications: backfill chunks cover hundreds of
        // blocks each, and shutdown drops the ExEx future before any final
        // persist — the periodic cadence is the only durable checkpoint.
        self.blocks_since_persist += tip - applied_from + 1;
        if self.blocks_since_persist >= PERSIST_EVERY {
            self.persist()?;
            self.blocks_since_persist = 0;
        }
        Ok(Some(B256::from(root)))
    }

    /// Roll back one committed block with a caller-supplied inverse (ExEx
    /// revert path for ranges the live path never saw).
    pub fn revert_committed(&mut self, number: u64, parent_hash: B256) -> anyhow::Result<()> {
        self.unwind_to(number - 1, parent_hash)
    }

    pub fn height(&self) -> u64 {
        self.height
    }

    pub fn head_hash(&self) -> B256 {
        self.head_hash
    }

    pub fn persist_now(&mut self) -> anyhow::Result<()> {
        self.persist()
    }
}

/// Map a `BundleState` to flat-MPT ops (hashed keys) — shared by the
/// validation hook and the ExEx feeder.
pub fn bundle_to_ops(bundle: &revm::database::BundleState) -> Vec<(Key, StateOp)> {
    let mut ops: Vec<(Key, StateOp)> = Vec::new();
    for (address, acct) in &bundle.state {
        let key: Key = keccak256(address.as_slice()).0;
        let destroyed = acct.status.was_destroyed();
        if destroyed {
            ops.push((key, StateOp::DeleteAccount));
        }
        match &acct.info {
            Some(info) => {
                ops.push((
                    key,
                    StateOp::SetAccount {
                        nonce: info.nonce,
                        balance: info.balance,
                        code_hash: info.code_hash.0,
                    },
                ));
            }
            None => {
                if !destroyed {
                    ops.push((key, StateOp::DeleteAccount));
                }
                continue;
            }
        }
        for (slot, value) in &acct.storage {
            let slot_key: Key = keccak256(slot.to_be_bytes::<32>()).0;
            let present = value.present_value();
            if present == U256::ZERO {
                ops.push((key, StateOp::DeleteStorage { slot: slot_key }));
            } else {
                ops.push((
                    key,
                    StateOp::SetStorage {
                        slot: slot_key,
                        value: mpt_flat_poc::eth::storage_value_rlp(present),
                    },
                ));
            }
        }
    }
    ops
}
