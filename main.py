def run_reliability(req: ReliabilityRequest) -> ReliabilityResponse:
    rng = np.random.default_rng(seed=42)  # seeded — reproducible

    # Demand — one uncertainty factor per load case (Turkstra combination)
    B_D = rng.lognormal(mean=np.log(1.0), sigma=req.demand_dead_cov, size=req.N)
    B_L = rng.gumbel(loc=1.0, scale=req.demand_live_cov, size=req.N)
    E = B_D * req.demand_dead_mean_kNm + B_L * req.demand_live_mean_kNm

    # Resistance — PC-girder flexure (verified formula)
    A_p = rng.normal(req.A_p_mean_mm2, req.A_p_mean_mm2 * req.A_p_cov, size=req.N)
    f_ps = rng.lognormal(mean=np.log(req.f_ps_mean_MPa), sigma=req.f_ps_cov, size=req.N)
    theta_R = rng.lognormal(mean=np.log(req.theta_R_mean), sigma=req.theta_R_cov, size=req.N)

    # Corrosion — two-phase degradation of tendon area
    loss_fraction = max(0.0, 1.0 - req.k_A * max(0.0, req.timeStep_years - req.T_i_years))
    loss_fraction = max(loss_fraction, 0.05)
    A_p_t = A_p * loss_fraction

    # Convert geometry from meters to millimeters — f_ck (MPa = N/mm2) and
    # f_ps (MPa) are naturally in mm-based units, so everything must match.
    b_f_mm = req.b_f_m * 1000
    d_p_mm = req.d_p_m * 1000

    T = A_p_t * f_ps                                   # tendon force, N (mm2 * MPa = N)
    a_mm = T / (0.85 * req.f_ck_MPa * b_f_mm)           # stress-block depth, mm
    z_mm = d_p_mm - a_mm / 2                             # lever arm, mm

    # N * mm -> kN * m: divide by 1e6 (1 kN.m = 1000 N * 1000 mm = 1e6 N.mm)
    R = theta_R * (T * z_mm) / 1e6                       # capacity, kN·m

    g = R - E
    n_fail = int(np.sum(g < 0))
    Pf = max(n_fail / req.N, 0.5 / req.N)
    beta = float(-norm.ppf(Pf))
    cov_Pf = float(np.sqrt((1 - Pf) / (Pf * req.N))) if Pf > 0 else None

    return ReliabilityResponse(
        timeStep_years=req.timeStep_years,
        N=req.N,
        n_fail=n_fail,
        Pf=Pf,
        beta=beta,
        cov_Pf=cov_Pf,
        mean_capacity_kNm=float(np.mean(R)),
        mean_demand_kNm=float(np.mean(E)),
    )
