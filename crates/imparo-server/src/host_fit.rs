//! Conservative auto-fit policy for the discrete-GPU KV Host tier.
//!
//! This module deliberately separates measurement from policy.  The backend owns the
//! device-transfer measurements, [`probe_disk_read`] measures the configured disk tier,
//! and [`derive_host_fit`] is a pure decision function.  Missing or zero facts never turn
//! into guessed defaults.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use imparo_backend::HostTierProfile;

static PROBE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const PROBE_ATTEMPTS: u64 = 32;

/// Machine-independent policy for deciding whether a discrete Host tier is worthwhile.
///
/// All quantities are operational policy, not model or GPU constants.  Callers may replace
/// the defaults without changing placement code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HostFitPolicy {
    /// A transfer must be at least `speed_margin_numerator / speed_margin_denominator`
    /// times the measured disk-read rate.  The ratio must be strictly greater than one.
    pub speed_margin_numerator: u64,
    pub speed_margin_denominator: u64,
    /// At most this fraction of currently available RAM may be assigned to the cache.
    pub max_available_ram_numerator: u64,
    pub max_available_ram_denominator: u64,
    /// RAM which must remain outside the Host tier even when the fraction permits more.
    pub retained_headroom_bytes: u64,
    /// Do not enable a cache too small to hold this many whole, unique content units.
    pub minimum_units: u64,
    /// Lower and upper bounds for the private sequential-read probe.
    pub probe_min_bytes: u64,
    pub probe_max_bytes: u64,
    /// Transfer-time horizon used to derive a representative probe size.
    pub probe_target_millis: u64,
    /// Maximum allocation used by the probe's streaming read/write buffer.
    pub probe_io_chunk_bytes: u64,
}

impl Default for HostFitPolicy {
    fn default() -> Self {
        Self {
            speed_margin_numerator: 3,
            speed_margin_denominator: 2,
            max_available_ram_numerator: 1,
            max_available_ram_denominator: 2,
            retained_headroom_bytes: 2_u64 << 30,
            minimum_units: 2,
            probe_min_bytes: 4_u64 << 20,
            probe_max_bytes: 64_u64 << 20,
            probe_target_millis: 20,
            probe_io_chunk_bytes: 1_u64 << 20,
        }
    }
}

/// A validated Host-tier allocation, expressed in whole unique content units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HostTierFit {
    pub capacity_bytes: u64,
    pub capacity_units: u64,
    pub unit_bytes: u64,
    pub measured_disk_read_bytes_per_second: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostFact {
    AvailableHostBytes,
    PinnedHostToDeviceRate,
    PinnedDeviceToHostRate,
    DiskReadRate,
    UnitBytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PolicyFault {
    SpeedMarginNotGreaterThanOne,
    RamFractionOutsideZeroToOne,
    ZeroMinimumUnits,
    InvalidProbeBounds,
    ZeroProbeTarget,
    ZeroProbeChunk,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArithmeticSite {
    RamFraction,
    SpeedMargin,
    ProbeSize,
    CapacityBytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiskProbeStage {
    InspectBase,
    Create,
    Write,
    Sync,
    OpenForRead,
    Read,
    Cleanup,
    Clock,
}

/// Structured fail-closed reasons suitable for startup diagnostics and tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostFitDisabled {
    UnknownOrZero(HostFact),
    DiskTierUnavailable,
    DiskBaseNotDirectory,
    InvalidPolicy(PolicyFault),
    TransferMarginNotMet {
        direction: TransferDirection,
        transfer_bytes_per_second: u64,
        required_bytes_per_second: u64,
    },
    InsufficientHeadroom {
        available_bytes: u64,
        retained_bytes: u64,
    },
    UserCapZero,
    CapacityBelowMinimum {
        capacity_units: u64,
        minimum_units: u64,
    },
    ArithmeticOverflow(ArithmeticSite),
    ProbeUnitExceedsMaximum,
    ProbeIo {
        stage: DiskProbeStage,
        kind: io::ErrorKind,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostFitDecision {
    Enabled(HostTierFit),
    Disabled(HostFitDisabled),
}

impl HostFitPolicy {
    fn validate(self) -> Result<(), HostFitDisabled> {
        if self.speed_margin_denominator == 0
            || self.speed_margin_numerator <= self.speed_margin_denominator
        {
            return Err(HostFitDisabled::InvalidPolicy(
                PolicyFault::SpeedMarginNotGreaterThanOne,
            ));
        }
        if self.max_available_ram_denominator == 0
            || self.max_available_ram_numerator == 0
            || self.max_available_ram_numerator > self.max_available_ram_denominator
        {
            return Err(HostFitDisabled::InvalidPolicy(
                PolicyFault::RamFractionOutsideZeroToOne,
            ));
        }
        if self.minimum_units == 0 {
            return Err(HostFitDisabled::InvalidPolicy(
                PolicyFault::ZeroMinimumUnits,
            ));
        }
        if self.probe_min_bytes == 0 || self.probe_max_bytes < self.probe_min_bytes {
            return Err(HostFitDisabled::InvalidPolicy(
                PolicyFault::InvalidProbeBounds,
            ));
        }
        if self.probe_target_millis == 0 {
            return Err(HostFitDisabled::InvalidPolicy(PolicyFault::ZeroProbeTarget));
        }
        if self.probe_io_chunk_bytes == 0 {
            return Err(HostFitDisabled::InvalidPolicy(PolicyFault::ZeroProbeChunk));
        }
        Ok(())
    }
}

/// Apply the Host-tier safety policy to already measured facts.
///
/// `user_cap_bytes` is the parsed value of an optional environment-style override.  A zero
/// cap explicitly disables the tier.  The returned byte capacity is always a checked whole
/// multiple of `unit_bytes`, so it counts unique units rather than conversation references.
pub(crate) fn derive_host_fit(
    profile: HostTierProfile,
    unit_bytes: u64,
    measured_disk_read_bytes_per_second: u64,
    user_cap_bytes: Option<u64>,
    policy: HostFitPolicy,
) -> HostFitDecision {
    if let Err(reason) = policy.validate() {
        return HostFitDecision::Disabled(reason);
    }
    for (value, fact) in [
        (profile.available_host_bytes, HostFact::AvailableHostBytes),
        (
            profile.pinned_h2d_bytes_per_second,
            HostFact::PinnedHostToDeviceRate,
        ),
        (
            profile.pinned_d2h_bytes_per_second,
            HostFact::PinnedDeviceToHostRate,
        ),
        (measured_disk_read_bytes_per_second, HostFact::DiskReadRate),
        (unit_bytes, HostFact::UnitBytes),
    ] {
        if value == 0 {
            return HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(fact));
        }
    }
    if user_cap_bytes == Some(0) {
        return HostFitDecision::Disabled(HostFitDisabled::UserCapZero);
    }

    let required =
        match required_transfer_rate(measured_disk_read_bytes_per_second, policy) {
            Ok(value) => value,
            Err(reason) => return HostFitDecision::Disabled(reason),
        };
    for (rate, direction) in [
        (
            profile.pinned_h2d_bytes_per_second,
            TransferDirection::HostToDevice,
        ),
        (
            profile.pinned_d2h_bytes_per_second,
            TransferDirection::DeviceToHost,
        ),
    ] {
        if rate < required {
            return HostFitDecision::Disabled(HostFitDisabled::TransferMarginNotMet {
                direction,
                transfer_bytes_per_second: rate,
                required_bytes_per_second: required,
            });
        }
    }

    if profile.available_host_bytes <= policy.retained_headroom_bytes {
        return HostFitDecision::Disabled(HostFitDisabled::InsufficientHeadroom {
            available_bytes: profile.available_host_bytes,
            retained_bytes: policy.retained_headroom_bytes,
        });
    }
    let fraction_budget = match mul_div_floor(
        profile.available_host_bytes,
        policy.max_available_ram_numerator,
        policy.max_available_ram_denominator,
        ArithmeticSite::RamFraction,
    ) {
        Ok(value) => value,
        Err(reason) => return HostFitDecision::Disabled(reason),
    };
    let headroom_budget = profile.available_host_bytes - policy.retained_headroom_bytes;
    let mut budget = fraction_budget.min(headroom_budget);
    if let Some(cap) = user_cap_bytes {
        budget = budget.min(cap);
    }
    let capacity_units = budget / unit_bytes;
    if capacity_units < policy.minimum_units {
        return HostFitDecision::Disabled(HostFitDisabled::CapacityBelowMinimum {
            capacity_units,
            minimum_units: policy.minimum_units,
        });
    }
    let Some(capacity_bytes) = capacity_units.checked_mul(unit_bytes) else {
        return HostFitDecision::Disabled(HostFitDisabled::ArithmeticOverflow(
            ArithmeticSite::CapacityBytes,
        ));
    };
    HostFitDecision::Enabled(HostTierFit {
        capacity_bytes,
        capacity_units,
        unit_bytes,
        measured_disk_read_bytes_per_second,
    })
}

/// Measure the configured disk tier and then apply [`derive_host_fit`].
///
/// The probe creates one private file directly below `disk_base`, synchronizes it before
/// timing a reopened sequential read, and removes it on success and every error path.  It
/// never opens, renames, or truncates a store-format file.
///
/// The OS may retain freshly written pages in its read cache.  That can only inflate the
/// disk rate and therefore biases this fail-closed policy toward disabling the Host tier.
pub(crate) fn probe_and_fit_host_tier(
    profile: HostTierProfile,
    unit_bytes: u64,
    disk_base: Option<&Path>,
    user_cap_bytes: Option<u64>,
    policy: HostFitPolicy,
) -> HostFitDecision {
    if user_cap_bytes == Some(0) {
        return HostFitDecision::Disabled(HostFitDisabled::UserCapZero);
    }
    let Some(disk_base) = disk_base else {
        return HostFitDecision::Disabled(HostFitDisabled::DiskTierUnavailable);
    };
    if let Err(reason) = validate_host_facts_before_probe(profile, unit_bytes, policy) {
        return HostFitDecision::Disabled(reason);
    }
    match probe_disk_read(disk_base, profile, unit_bytes, policy) {
        Ok(rate) => derive_host_fit(profile, unit_bytes, rate, user_cap_bytes, policy),
        Err(reason) => HostFitDecision::Disabled(reason),
    }
}

fn validate_host_facts_before_probe(
    profile: HostTierProfile,
    unit_bytes: u64,
    policy: HostFitPolicy,
) -> Result<(), HostFitDisabled> {
    policy.validate()?;
    for (value, fact) in [
        (profile.available_host_bytes, HostFact::AvailableHostBytes),
        (
            profile.pinned_h2d_bytes_per_second,
            HostFact::PinnedHostToDeviceRate,
        ),
        (
            profile.pinned_d2h_bytes_per_second,
            HostFact::PinnedDeviceToHostRate,
        ),
        (unit_bytes, HostFact::UnitBytes),
    ] {
        if value == 0 {
            return Err(HostFitDisabled::UnknownOrZero(fact));
        }
    }
    Ok(())
}

fn required_transfer_rate(
    disk_rate: u64,
    policy: HostFitPolicy,
) -> Result<u64, HostFitDisabled> {
    let product = u128::from(disk_rate) * u128::from(policy.speed_margin_numerator);
    let denominator = u128::from(policy.speed_margin_denominator);
    let required = product.div_ceil(denominator);
    u64::try_from(required)
        .map_err(|_| HostFitDisabled::ArithmeticOverflow(ArithmeticSite::SpeedMargin))
}

fn mul_div_floor(
    value: u64,
    numerator: u64,
    denominator: u64,
    site: ArithmeticSite,
) -> Result<u64, HostFitDisabled> {
    let result = u128::from(value) * u128::from(numerator) / u128::from(denominator);
    u64::try_from(result).map_err(|_| HostFitDisabled::ArithmeticOverflow(site))
}

fn probe_sample_bytes(
    profile: HostTierProfile,
    unit_bytes: u64,
    policy: HostFitPolicy,
) -> Result<u64, HostFitDisabled> {
    let maximum_units = policy.probe_max_bytes / unit_bytes;
    if maximum_units == 0 {
        return Err(HostFitDisabled::ProbeUnitExceedsMaximum);
    }
    let lower_bytes = policy.probe_min_bytes.max(unit_bytes);
    let minimum_units = lower_bytes.div_ceil(unit_bytes).min(maximum_units);
    let slow_transfer = profile
        .pinned_h2d_bytes_per_second
        .min(profile.pinned_d2h_bytes_per_second);
    let target_bytes = u128::from(slow_transfer)
        .checked_mul(u128::from(policy.probe_target_millis))
        .ok_or(HostFitDisabled::ArithmeticOverflow(
            ArithmeticSite::ProbeSize,
        ))?
        / 1_000;
    let target_units = target_bytes.div_ceil(u128::from(unit_bytes));
    let target_units = u64::try_from(target_units)
        .map_err(|_| HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize))?;
    let units = target_units.clamp(minimum_units, maximum_units);
    units
        .checked_mul(unit_bytes)
        .ok_or(HostFitDisabled::ArithmeticOverflow(
            ArithmeticSite::ProbeSize,
        ))
}

/// Sequentially measure reads from a private, durable probe file below `disk_base`.
pub(crate) fn probe_disk_read(
    disk_base: &Path,
    profile: HostTierProfile,
    unit_bytes: u64,
    policy: HostFitPolicy,
) -> Result<u64, HostFitDisabled> {
    validate_host_facts_before_probe(profile, unit_bytes, policy)?;
    let metadata =
        std::fs::metadata(disk_base).map_err(|error| HostFitDisabled::ProbeIo {
            stage: DiskProbeStage::InspectBase,
            kind: error.kind(),
        })?;
    if !metadata.is_dir() {
        return Err(HostFitDisabled::DiskBaseNotDirectory);
    }
    let sample_bytes = probe_sample_bytes(profile, unit_bytes, policy)?;
    let chunk_bytes = policy.probe_io_chunk_bytes.min(sample_bytes);
    let chunk_len = usize::try_from(chunk_bytes)
        .map_err(|_| HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize))?;
    let mut temp = PrivateProbeFile::create(disk_base)?;
    let mut buffer = vec![0xA5_u8; chunk_len];
    let mut remaining = sample_bytes;
    while remaining != 0 {
        let take = remaining.min(chunk_bytes);
        let take = usize::try_from(take).map_err(|_| {
            HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize)
        })?;
        temp.file_mut()
            .write_all(&buffer[..take])
            .map_err(|error| HostFitDisabled::ProbeIo {
                stage: DiskProbeStage::Write,
                kind: error.kind(),
            })?;
        remaining -= u64::try_from(take).map_err(|_| {
            HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize)
        })?;
    }
    temp.file_mut()
        .sync_all()
        .map_err(|error| HostFitDisabled::ProbeIo {
            stage: DiskProbeStage::Sync,
            kind: error.kind(),
        })?;
    temp.close_file();
    temp.open_for_read()?;

    let started = Instant::now();
    let mut remaining = sample_bytes;
    while remaining != 0 {
        let take = remaining.min(chunk_bytes);
        let take = usize::try_from(take).map_err(|_| {
            HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize)
        })?;
        temp.file_mut()
            .read_exact(&mut buffer[..take])
            .map_err(|error| HostFitDisabled::ProbeIo {
                stage: DiskProbeStage::Read,
                kind: error.kind(),
            })?;
        remaining -= u64::try_from(take).map_err(|_| {
            HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize)
        })?;
    }
    let elapsed = started.elapsed();
    temp.finish()?;
    let nanos = elapsed.as_nanos();
    if nanos == 0 {
        return Err(HostFitDisabled::ProbeIo {
            stage: DiskProbeStage::Clock,
            kind: io::ErrorKind::Other,
        });
    }
    let rate = u128::from(sample_bytes) * 1_000_000_000_u128 / nanos;
    let rate = u64::try_from(rate)
        .map_err(|_| HostFitDisabled::ArithmeticOverflow(ArithmeticSite::ProbeSize))?;
    if rate == 0 {
        return Err(HostFitDisabled::UnknownOrZero(HostFact::DiskReadRate));
    }
    Ok(rate)
}

struct PrivateProbeFile {
    path: Option<PathBuf>,
    file: Option<File>,
}

impl PrivateProbeFile {
    fn create(base: &Path) -> Result<Self, HostFitDisabled> {
        for _ in 0..PROBE_ATTEMPTS {
            let sequence = PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!(
                ".imparo-host-fit-probe-{}-{sequence}.tmp",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path: Some(path),
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(HostFitDisabled::ProbeIo {
                        stage: DiskProbeStage::Create,
                        kind: error.kind(),
                    });
                }
            }
        }
        Err(HostFitDisabled::ProbeIo {
            stage: DiskProbeStage::Create,
            kind: io::ErrorKind::AlreadyExists,
        })
    }

    fn file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("private probe file is open in this phase")
    }

    fn close_file(&mut self) {
        drop(self.file.take());
    }

    fn open_for_read(&mut self) -> Result<(), HostFitDisabled> {
        let path = self.path.as_ref().expect("private probe path exists");
        self.file =
            Some(File::open(path).map_err(|error| HostFitDisabled::ProbeIo {
                stage: DiskProbeStage::OpenForRead,
                kind: error.kind(),
            })?);
        Ok(())
    }

    fn finish(mut self) -> Result<(), HostFitDisabled> {
        self.close_file();
        let path = self.path.as_ref().expect("private probe path exists");
        std::fs::remove_file(path).map_err(|error| HostFitDisabled::ProbeIo {
            stage: DiskProbeStage::Cleanup,
            kind: error.kind(),
        })?;
        self.path = None;
        Ok(())
    }
}

impl Drop for PrivateProbeFile {
    fn drop(&mut self) {
        self.close_file();
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(available: u64, h2d: u64, d2h: u64) -> HostTierProfile {
        HostTierProfile {
            available_host_bytes: available,
            pinned_h2d_bytes_per_second: h2d,
            pinned_d2h_bytes_per_second: d2h,
        }
    }

    fn policy() -> HostFitPolicy {
        HostFitPolicy {
            speed_margin_numerator: 3,
            speed_margin_denominator: 2,
            max_available_ram_numerator: 1,
            max_available_ram_denominator: 2,
            retained_headroom_bytes: 1_000,
            minimum_units: 2,
            probe_min_bytes: 4_096,
            probe_max_bytes: 65_536,
            probe_target_millis: 1,
            probe_io_chunk_bytes: 4_096,
        }
    }

    #[test]
    fn unknown_or_zero_facts_fail_closed() {
        assert_eq!(
            derive_host_fit(profile(0, 300, 300), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(
                HostFact::AvailableHostBytes
            ))
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 0, 300), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(
                HostFact::PinnedHostToDeviceRate
            ))
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 300, 0), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(
                HostFact::PinnedDeviceToHostRate
            ))
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 300, 300), 100, 0, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(
                HostFact::DiskReadRate
            ))
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 300, 300), 0, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::UnknownOrZero(
                HostFact::UnitBytes
            ))
        );
    }

    #[test]
    fn both_transfer_directions_must_meet_the_exact_margin_boundary() {
        assert!(matches!(
            derive_host_fit(profile(10_000, 150, 150), 100, 100, None, policy()),
            HostFitDecision::Enabled(_)
        ));
        assert_eq!(
            derive_host_fit(profile(10_000, 149, 200), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::TransferMarginNotMet {
                direction: TransferDirection::HostToDevice,
                transfer_bytes_per_second: 149,
                required_bytes_per_second: 150,
            })
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 200, 149), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::TransferMarginNotMet {
                direction: TransferDirection::DeviceToHost,
                transfer_bytes_per_second: 149,
                required_bytes_per_second: 150,
            })
        );
    }

    #[test]
    fn retained_headroom_and_fraction_both_limit_capacity() {
        assert_eq!(
            derive_host_fit(profile(1_000, 300, 300), 100, 100, None, policy()),
            HostFitDecision::Disabled(HostFitDisabled::InsufficientHeadroom {
                available_bytes: 1_000,
                retained_bytes: 1_000,
            })
        );
        let HostFitDecision::Enabled(fit) =
            derive_host_fit(profile(10_000, 300, 300), 700, 100, None, policy())
        else {
            panic!("expected an enabled fit");
        };
        assert_eq!(fit.capacity_units, 7);
        assert_eq!(fit.capacity_bytes, 4_900);
    }

    #[test]
    fn cap_clamps_to_whole_unique_units_and_can_disable() {
        let HostFitDecision::Enabled(fit) =
            derive_host_fit(profile(10_000, 300, 300), 700, 100, Some(2_099), policy())
        else {
            panic!("expected an enabled fit");
        };
        assert_eq!(fit.capacity_units, 2);
        assert_eq!(fit.capacity_bytes, 1_400);
        assert_eq!(
            derive_host_fit(profile(10_000, 300, 300), 700, 100, Some(0), policy()),
            HostFitDecision::Disabled(HostFitDisabled::UserCapZero)
        );
        assert_eq!(
            derive_host_fit(profile(10_000, 300, 300), 700, 100, Some(1_399), policy(),),
            HostFitDecision::Disabled(HostFitDisabled::CapacityBelowMinimum {
                capacity_units: 1,
                minimum_units: 2,
            })
        );
    }

    #[test]
    fn checked_ratio_rejects_an_unrepresentable_required_rate() {
        let mut p = policy();
        p.speed_margin_numerator = u64::MAX;
        p.speed_margin_denominator = 1;
        assert_eq!(
            derive_host_fit(
                profile(u64::MAX, u64::MAX, u64::MAX),
                1,
                u64::MAX,
                None,
                p,
            ),
            HostFitDecision::Disabled(HostFitDisabled::ArithmeticOverflow(
                ArithmeticSite::SpeedMargin
            ))
        );
    }

    #[test]
    fn missing_disk_path_disables_without_probing() {
        assert_eq!(
            probe_and_fit_host_tier(
                profile(10_000, 300, 300),
                100,
                None,
                None,
                policy()
            ),
            HostFitDecision::Disabled(HostFitDisabled::DiskTierUnavailable)
        );
    }

    #[test]
    fn zero_user_cap_disables_without_a_disk_probe() {
        assert_eq!(
            probe_and_fit_host_tier(
                profile(10_000, 300, 300),
                100,
                None,
                Some(0),
                policy()
            ),
            HostFitDecision::Disabled(HostFitDisabled::UserCapZero)
        );
    }

    #[test]
    fn disk_probe_is_positive_and_leaves_no_private_file() {
        let directory = std::env::temp_dir().join(format!(
            "imparo-host-fit-test-{}-{}",
            std::process::id(),
            PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).expect("create private test directory");
        let rate = probe_disk_read(
            &directory,
            profile(10_000, 64 << 20, 64 << 20),
            4_096,
            policy(),
        )
        .expect("probe succeeds");
        assert!(rate > 0);
        assert_eq!(
            std::fs::read_dir(&directory)
                .expect("read test directory")
                .count(),
            0
        );
        std::fs::remove_dir(&directory).expect("remove private test directory");
    }

    #[test]
    fn probe_rejects_a_unit_larger_than_its_bound_without_creating_a_file() {
        let directory = std::env::temp_dir().join(format!(
            "imparo-host-fit-bound-test-{}-{}",
            std::process::id(),
            PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).expect("create private test directory");
        assert_eq!(
            probe_disk_read(
                &directory,
                profile(10_000, 64 << 20, 64 << 20),
                policy().probe_max_bytes + 1,
                policy(),
            ),
            Err(HostFitDisabled::ProbeUnitExceedsMaximum)
        );
        assert_eq!(
            std::fs::read_dir(&directory)
                .expect("read test directory")
                .count(),
            0
        );
        std::fs::remove_dir(&directory).expect("remove private test directory");
    }
}
