"""
The exp_propagator and L_storage input options: valid combinations, the pre-1.3.0 aliases
Taylor_sparse and Krylov_sparse, and the errors for combinations that do not exist.
"""
import pytest

from cases import set_up_case

CASE = "1spin_linear_Taylor"   # short and small; only the set-up is exercised here


def options(**kw):
    return {2: kw, 3: {"save_drdt": False}}


@pytest.mark.parametrize("kw, name, storage, sparse", [
    ({"exp_propagator": "Pade"},                          "Pade",   "dense",  False),
    ({"exp_propagator": "Taylor"},                        "Taylor", "dense",  False),
    ({"exp_propagator": "Krylov", "L_storage": "dense"},  "Krylov", "dense",  False),
    ({"exp_propagator": "Taylor", "L_storage": "sparse"}, "Taylor", "sparse", True),
    ({"exp_propagator": "Krylov", "L_storage": "sparse"}, "Krylov", "sparse", True),
    # Pre-1.3.0 names, still accepted
    ({"exp_propagator": "Taylor_sparse"},                 "Taylor", "sparse", True),
    ({"exp_propagator": "Krylov_sparse", "L_storage": "sparse"}, "Krylov", "sparse", True),
])
def test_valid(kw, name, storage, sparse, workdir):
    lio = set_up_case(CASE, workdir, extra=options(**kw))
    assert (lio.exp_propagator, lio.L_storage, lio.sparse_L) == (name, storage, sparse)
    assert lio.propagate_risvrho.__name__ == "propagate_risvrho_" + name + ("_sparse" if sparse else "")


def test_legacy_name_uses_same_propagator(workdir):
    a = set_up_case(CASE, workdir / "a", extra=options(exp_propagator="Krylov_sparse"))
    b = set_up_case(CASE, workdir / "b", extra=options(exp_propagator="Krylov", L_storage="sparse"))
    assert a.propagate_risvrho.__name__ == b.propagate_risvrho.__name__


@pytest.mark.parametrize("kw, message", [
    ({"exp_propagator": "Pade", "L_storage": "sparse"},          "needs L_storage = dense"),
    ({"exp_propagator": "Strang", "L_storage": "sparse"},        "forms no L"),
    ({"exp_propagator": "Taylor_sparse", "L_storage": "dense"},  "implies L_storage = sparse"),
    ({"exp_propagator": "Taylor", "L_storage": "csr"},           "Invalid L_storage"),
    ({"exp_propagator": "Expm"},                                 "Invalid exp_propagator"),
])
def test_invalid(kw, message, workdir):
    with pytest.raises(ValueError, match=message):
        set_up_case(CASE, workdir, extra=options(**kw))


def test_rk4_rejects_sparse(workdir):
    lio = set_up_case(CASE, workdir, extra=options(exp_propagator="Taylor", L_storage="sparse", method="RK4"))
    with pytest.raises(ValueError, match="requires a dense L"):
        lio.evolve_rho()


@pytest.mark.parametrize("save, tmax, message", [
    # 10 stairs of 5 ps, but magnetization saved every 20 stairs: no save fits
    ({"save_mag": True, "nt_mag": 20, "save_rho": False}, 50.0, "save intervals of nt_mag = 20 stairs"),
    # 10.4 stairs and nothing saved
    ({"save_mag": False, "save_rho": False},              52.0, "10.4 stairs"),
])
def test_loop_mismatch_message(save, tmax, message, workdir):
    lio = set_up_case(CASE, workdir, extra=options(exp_propagator="Taylor"))
    for k, v in save.items():
        setattr(lio, k, v)
    lio.save_drdt, lio.tmax = False, tmax
    with pytest.raises(ValueError, match=message):
        lio.set_up_loop()


def test_loop_whole_save_intervals(workdir):
    lio = set_up_case(CASE, workdir, extra=options(exp_propagator="Taylor"))
    lio.save_mag, lio.nt_mag, lio.save_rho, lio.save_drdt = True, 5, False, False
    lio.set_up_loop()
    assert lio.nt == 10
