#![no_main]

use ed25519_dalek::SigningKey;
use imparo_program_pack::{DistributionScope, TrustStore, TrustedKey};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let key = SigningKey::from_bytes(&[42; 32]);
    let mut trust = TrustStore::new();
    trust.add(TrustedKey {
        key_id: "imparo.fuzz.community".into(),
        public_key: key.verifying_key().to_bytes(),
        scope: DistributionScope::Community,
        release_channel: "imparo.community.stable".into(),
    }).expect("fixed fuzz trust root is valid");
    let split = data.len().min(1024);
    let (manifest, envelope) = data.split_at(split);
    let _ = trust.verify(
        manifest,
        envelope,
        DistributionScope::Community,
        "imparo.community.stable",
    );
});
