"""
Structural Reliability Service
================================
FastAPI microservice that computes structural reliability (Pf, beta)
via direct Monte Carlo simulation, using verified closed-form capacity
and demand formulas.

Endpoints:
  POST /reliability - run direct Monte Carlo reliability analysis
  GET  /health       - health check
"""

import numpy as np
from scipy.stats import norm
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Structural Reliability Service",
    description=(
        "Physics-guided direct Monte Carlo reliability analysis. "
        "Computes Pf(t) and beta(t) from verified capacity/demand formulas."
    ),
    version="2.0.0",
)


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------

class ReliabilityRequest(BaseModel):
    N: int
    timeStep_years: float

    # Demand — from structural analysis (MIDAS/SAP2000), per load case
    demand_dead_mean_kNm: float
    demand_live_mean_kNm: float
    demand_dead_cov: float
    demand_live_cov: float

    # Resistance — PC-girder flexure (AASHTO 5.6.3 / EN 6.1)
    A_p_mean_mm2: float
    A_p_cov: float
    f_ps_mean_MPa: float
    f_ps_cov: float
    f_ck_MPa: float
    b_f_m: float
    d_p_m: float
    theta_R_mean: float
    theta_R_cov: float

    # Corrosion — two-phase (fib Bulletin 34/59)
    T_i_years: float
    k_A: float


class ReliabilityResponse(BaseModel):
    timeStep_years: float
    N: int
    n_fail: int
    Pf: float
    beta: float
    cov_Pf: Optional[float]
    mean_capacity_kNm: float
    mean_demand_kNm: float


# --------------------------------------------------------------------------
# Core Monte Carlo function
# --------------------------------------------------------------------------

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

    T = A_p_t * f_ps                                        # tendon force, N
    a = T / (0.85 * req.f_ck_MPa * req.b_f_m )         # stress-block depth, mm
    z = req.d_p_m - a / 2                                     # lever arm, mm
   # N * mm -> kN * m: divide by 1e6 (1 kN.m = 1000 N * 1000 mm = 1e6 N.mm)
    R = theta_R * (T * z_mm) / 1e6                       # kN·m

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


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reliability", response_model=ReliabilityResponse)
def reliability_endpoint(req: ReliabilityRequest):
    return run_reliability(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
