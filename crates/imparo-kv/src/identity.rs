//! Content identity: the rule everything follows from.
//!
//! A unit's identity is a hash chained over the entire prefix, rooted in the
//! configuration that determines the numbers:
//!
//! ```text
//! root    = H( model digest || kv dtype/quant || rope config || layer geometry )
//! hash[0] = H( root       || tokens of unit 0 )
//! hash[i] = H( hash[i-1]  || tokens of unit i )
//! ```
//!
//! Chained, because K/V values depend on every preceding token. Rooted, because a
//! different model or KV quantisation produces different bytes for the same tokens.
//! The property this buys: EQUAL HASH IMPLIES EQUAL BYTES — sharing needs no
//! validation on borrow. The hash is over tokens, so it is identical on every
//! backend and every machine; nothing tuned may enter it.

use sha2::{Digest, Sha256};

/// Tokens per content unit. A FORMAT constant, not a tuned knob: it defines hash
/// identity, and hash identity must not depend on which machine wrote the bytes.
/// Chosen generously as a power of two every plausible kernel tile divides; changing
/// it is a format bump (manifest version) that ages out the old store.
pub const UNIT_TOKENS: usize = 256;

/// 128-bit unit hash (SHA-256 truncated). Wide enough that collision is not a case
/// the code handles — no byte-verify on hit, no guard path (design: "Hash width").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct UnitHash(pub [u8; 16]);

impl UnitHash {
    #[must_use]
    /// The inverse of `hex`: 32 hex characters back to a hash, or None.
    ///
    /// Beside `hex` on purpose -- a spelling and its parser that live apart drift
    /// apart, and this pair is what a manifest line means.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 32 {
            return None;
        }
        let mut b = [0_u8; 16];
        for (i, c) in s.as_bytes().chunks_exact(2).enumerate() {
            b[i] = u8::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok()?;
        }
        Some(Self(b))
    }

    pub fn hex(&self) -> String {
        use std::fmt::Write as _;
        self.0.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    }
}

/// The configuration root. Two runs share KV bytes only if every input that shapes
/// those bytes is identical; everything that shapes them goes in here, and nothing
/// else does (a tuned tile width does NOT shape the values, so it must not enter).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigRoot([u8; 32]);

impl ConfigRoot {
    /// The root's bytes, for scoping the on-disk store directory.
    #[must_use]
    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// `model_digest`: the GGUF's content identity (imparo-gguf snapshot sha256 or
    /// equivalent). `kv_types`: the K and V cache wire types. `rope_and_geometry`:
    /// the plan's serialized per-layer geometry (rope config, head counts, state
    /// kinds) — the plan owns what belongs in it; this module only hashes bytes.
    #[must_use]
    pub fn new(
        model_digest: &[u8],
        kv_types: (u32, u32),
        rope_and_geometry: &[u8],
    ) -> Self {
        let mut h = Sha256::new();
        h.update(b"imparo-kv-root-v1");
        h.update((model_digest.len() as u64).to_le_bytes());
        h.update(model_digest);
        h.update(kv_types.0.to_le_bytes());
        h.update(kv_types.1.to_le_bytes());
        h.update((rope_and_geometry.len() as u64).to_le_bytes());
        h.update(rope_and_geometry);
        Self(h.finalize().into())
    }
}

/// Chains the unit hashes for a token stream. Only WHOLE units are hashed — the
/// tail partial unit is unshareable by design ("do not share partial units") and
/// gets no identity.
#[must_use]
pub fn unit_hashes(root: &ConfigRoot, tokens: &[u32]) -> Vec<UnitHash> {
    let mut out = Vec::with_capacity(tokens.len() / UNIT_TOKENS);
    let mut prev: [u8; 32] = root.0;
    for unit in tokens.chunks_exact(UNIT_TOKENS) {
        let mut h = Sha256::new();
        h.update(prev);
        for t in unit {
            h.update(t.to_le_bytes());
        }
        let full: [u8; 32] = h.finalize().into();
        let mut short = [0_u8; 16];
        short.copy_from_slice(&full[..16]);
        out.push(UnitHash(short));
        prev = full;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> ConfigRoot {
        ConfigRoot::new(b"model-A", (1, 1), b"geom-1")
    }

    #[test]
    fn shared_prefix_shares_hashes_divergence_splits_them() {
        let mut a: Vec<u32> = (0..3 * UNIT_TOKENS as u32).collect();
        let mut b = a.clone();
        // diverge inside unit 2
        b[2 * UNIT_TOKENS + 7] ^= 1;
        let (ha, hb) = (unit_hashes(&root(), &a), unit_hashes(&root(), &b));
        assert_eq!(ha[0], hb[0]);
        assert_eq!(ha[1], hb[1]);
        assert_ne!(ha[2], hb[2]);
        // and the chain makes EVERYTHING after an early divergence differ,
        // even where the unit's own tokens are equal again
        a[3] ^= 1;
        let ha2 = unit_hashes(&root(), &a);
        assert_ne!(ha2[0], hb[0]);
        assert_ne!(ha2[1], hb[1]);
        assert_ne!(ha2[2], hb[2]);
    }

    #[test]
    fn root_inputs_split_identity() {
        let t: Vec<u32> = (0..UNIT_TOKENS as u32).collect();
        let base = unit_hashes(&ConfigRoot::new(b"m", (1, 1), b"g"), &t);
        for other in [
            ConfigRoot::new(b"m2", (1, 1), b"g"),
            ConfigRoot::new(b"m", (2, 1), b"g"),
            ConfigRoot::new(b"m", (1, 2), b"g"),
            ConfigRoot::new(b"m", (1, 1), b"g2"),
        ] {
            assert_ne!(base[0], unit_hashes(&other, &t)[0]);
        }
    }

    #[test]
    fn partial_tail_gets_no_identity() {
        let t: Vec<u32> = (0..(UNIT_TOKENS as u32 * 2) + 5).collect();
        assert_eq!(unit_hashes(&root(), &t).len(), 2);
    }
}
