use anyhow::Ok;
use cast::cost_model::FusionConfig;
use cast::cpu::CPUKernelGenSpec;

fn main() -> anyhow::Result<()> {
    let profile = cast::cpu::measure_cpu_profile(&CPUKernelGenSpec::f64())?;
    let config = FusionConfig::hardware_adaptive(&profile, /*max_size=*/ 6);

    println!(
        "Profile: crossover_ai={:.6}, peak_bw_gib_s={:.6}",
        profile.crossover_ai, profile.peak_bw_gib_s
    );
    println!(
        "Config: size_max={}, benefit_margin={:.6}",
        config.size_max, config.benefit_margin
    );

    Ok(())
}
