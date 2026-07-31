import shutil, os
import numpy as np
from itertools import product
from scipy.interpolate import griddata
import ROOT
from hepdata_lib import Submission, Table, Variable, RootFileReader, Uncertainty

shutil.rmtree("output", ignore_errors=True)
if os.path.exists("submission.tar.gz"):
    os.remove("submission.tar.gz")

submission = Submission()

def _read_tgraph(root_file, key_name):
    f = ROOT.TFile(root_file)
    tgraph = f.GetKey(key_name).ReadObj()
    n = tgraph.GetN()
    return [tgraph.GetPointX(i) for i in range(n)], [tgraph.GetPointY(i) for i in range(n)]


def make_table_two_1Dscans(root_file_1, key_1, label_1, root_file_2, key_2, label_2, table_name, description, location, image=None, x_range=None):
    xs, ys1 = _read_tgraph(root_file_1, key_1)
    _,  ys2 = _read_tgraph(root_file_2, key_2)

    if x_range is not None:
        mask = [x_range[0] <= x <= x_range[1] for x in xs]
        xs  = [x for x, m in zip(xs,  mask) if m]
        ys1 = [y for y, m in zip(ys1, mask) if m]
        ys2 = [y for y, m in zip(ys2, mask) if m]

    alpha = Variable(r"$\alpha^{ \mathrm{H} \tau \tau}$", is_independent=True, is_binned=False, units="degrees")
    alpha.values = xs

    nll1 = Variable(label_1, is_independent=False, is_binned=False, units="")
    nll1.values = [y for y in ys1]

    nll2 = Variable(label_2, is_independent=False, is_binned=False, units="")
    nll2.values = [y for y in ys2]

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(alpha)
    table.add_variable(nll1)
    table.add_variable(nll2)
    return table


def make_table_1Dscan(root_file, key_name, table_name, description, location, image=None):
    xs, ys = _read_tgraph(root_file, key_name)

    alpha = Variable(r"$\alpha^{ \mathrm{H} \tau \tau}$", is_independent=True, is_binned=False, units="degrees")
    alpha.values = xs

    nll = Variable(r"$-2 \Delta \ln L$", is_independent=False, is_binned=False, units="")
    nll.values = ys

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(alpha)
    table.add_variable(nll)
    return table



def make_table_phiCP(root_file, directory, table_name, description, location, image=None):
    reader = RootFileReader(root_file)

    def _read(key):
        return reader.read_hist_1d(f"{directory}/{key}")

    hist_sm   = _read("sig_sm")
    hist_ps   = _read("sig_ps")
    hist_mm   = _read("sig_mm")
    hist_bkg  = _read("bkg")
    hist_data = _read("data_obs")

    phi = Variable(r"$\phi_{CP}$", is_independent=True, is_binned=True, units="degrees")
    phi.values = hist_sm["x_edges"]

    def _make_var(label, hist):
        var = Variable(label, is_independent=False, is_binned=False, units="")
        var.values = hist["y"]
        if any(dy != 0 for dy in hist["dy"]):
            unc = Uncertainty("", is_symmetric=True)
            unc.values = hist["dy"]
            var.add_uncertainty(unc)
        return var

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(phi)
    table.add_variable(_make_var("Data-Background", hist_data))
    table.add_variable(_make_var("Background uncertainty", hist_bkg))
    table.add_variable(_make_var(r"$\alpha^{\mathrm{H}\tau\tau}=0^\circ$", hist_sm))
    table.add_variable(_make_var(r"$\alpha^{\mathrm{H}\tau\tau}=90^\circ$", hist_ps))
    table.add_variable(_make_var(r"$\alpha^{\mathrm{H}\tau\tau}=45^\circ$", hist_mm))
    return table


def make_table_fit_inputs(root_file, directory, x_label, x_units, table_name, description, location, image=None):
    reader = RootFileReader(root_file)

    def _read(key):
        return reader.read_hist_1d(f"{directory}/{key}")

    hist_data = _read("data_obs")
    hist_bkg  = _read("TotalBkg")
    hist_sig  = _read("TotalSig")
    hist_sm   = _read("TotalSig_SM")
    hist_ps   = _read("TotalSig_PS")
    _other_hists   = [_read(k) for k in ("ZL", "VVT", "TTT")]
    hist_other     = {**_other_hists[0], "y": [sum(h["y"][i] for h in _other_hists) for i in range(len(_other_hists[0]["y"]))]}
    _jetfake_hists = [_read(k) for k in ("JetFakes", "JetFakesSublead")]
    hist_jetfake   = {**_jetfake_hists[0], "y": [sum(h["y"][i] for h in _jetfake_hists) for i in range(len(_jetfake_hists[0]["y"]))]}
    hist_ztt       = _read("ZTT")

    x_var = Variable(x_label, is_independent=True, is_binned=True, units=x_units)
    x_var.values = hist_data["x_edges"]

    def _make_var(label, hist):
        var = Variable(label, is_independent=False, is_binned=False, units="")
        var.values = hist["y"]
        if any(dy != 0 for dy in hist["dy"]):
            unc = Uncertainty("", is_symmetric=True)
            unc.values = hist["dy"]
            var.add_uncertainty(unc)
        return var

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(x_var)
    table.add_variable(_make_var("Data",             hist_data))
    table.add_variable(_make_var("Total Background", hist_bkg))
    table.add_variable(_make_var(r"Z$\to\tau\tau$",   hist_ztt))
    table.add_variable(_make_var(r"Jet$\to\tau_{h}$", hist_jetfake))
    table.add_variable(_make_var("Other",            hist_other))
    table.add_variable(_make_var(r"H$\to\tau\tau$ (Best Fit)", hist_sig))
    table.add_variable(_make_var(r"H$\to\tau\tau$ ($\alpha^{\mathrm{H}\tau\tau}=0^\circ$)",  hist_sm))
    table.add_variable(_make_var(r"H$\to\tau\tau$ ($\alpha^{\mathrm{H}\tau\tau}=90^\circ$)", hist_ps))
    return table


def make_table_bdt_score(root_file, directory, table_name, description, location, image=None):
    reader = RootFileReader(root_file)

    def _read(key):
        return reader.read_hist_1d(f"{directory}/{key}")

    hist_data    = _read("data_obs")
    hist_bkg     = _read("TotalBkg")

    _other_hists = [_read(k) for k in ("ZL", "VVT", "TTT")]
    hist_other   = {**_other_hists[0], "y": [sum(h["y"][i] for h in _other_hists) for i in range(len(_other_hists[0]["y"]))]}
    _jetfake_hists = [_read(k) for k in ("JetFakes", "JetFakesSublead")]
    hist_jetfake   = {**_jetfake_hists[0], "y": [sum(h["y"][i] for h in _jetfake_hists) for i in range(len(_jetfake_hists[0]["y"]))]}
    hist_ztt     = _read("ZTT")

    x_var = Variable("BDT score", is_independent=True, is_binned=True, units="")
    x_var.values = hist_data["x_edges"]

    def _make_var(label, hist):
        var = Variable(label, is_independent=False, is_binned=False, units="")
        var.values = hist["y"]
        if any(dy != 0 for dy in hist["dy"]):
            unc = Uncertainty("", is_symmetric=True)
            unc.values = hist["dy"]
            var.add_uncertainty(unc)
        return var

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(x_var)
    table.add_variable(_make_var("Data",             hist_data))
    table.add_variable(_make_var("Total Background", hist_bkg))
    table.add_variable(_make_var(r"Z$\to\tau\tau$",   hist_ztt))
    table.add_variable(_make_var(r"Jet$\to\tau_{h}$", hist_jetfake))
    table.add_variable(_make_var("Other",            hist_other))
    return table


def _fill_missing_grid_points(xs, ys, nll_vals):
    # Detect missing points on 2D scans and interpolate NLL there
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)

    existing = set(zip(np.round(xs, 8), np.round(ys, 8)))
    missing_x, missing_y = [], []
    for gx, gy in product(unique_x, unique_y):
        if (round(gx, 8), round(gy, 8)) not in existing:
            missing_x.append(gx)
            missing_y.append(gy)

    if not missing_x:
        return xs, ys, nll_vals

    points = np.column_stack([xs, ys])
    interp_nll = griddata(points, nll_vals, (missing_x, missing_y), method="cubic")

    # Discard any NaN (precaution)
    mask = np.isfinite(interp_nll)
    filled_x = np.concatenate([xs, np.array(missing_x)[mask]])
    filled_y = np.concatenate([ys, np.array(missing_y)[mask]])
    filled_nll = np.concatenate([nll_vals, interp_nll[mask]])

    n_filled = mask.sum()
    n_missing = len(missing_x)
    print(f"  Interpolated {n_filled}/{n_missing} missing grid points ({n_missing - n_filled} outside convex hull dropped)")
    return filled_x, filled_y, filled_nll


def make_table_correlation_matrix(params, matrix, table_name, description, location, image=None):
    rows, cols, vals = [], [], []
    for i, row_p in enumerate(params):
        for j, col_p in enumerate(params):
            if j <= i:
                rows.append(row_p)
                cols.append(col_p)
                vals.append(matrix[i][j])

    var_row = Variable("Parameter (row)", is_independent=True, is_binned=False)
    var_row.values = rows

    var_col = Variable("Parameter (column)", is_independent=True, is_binned=False)
    var_col.values = cols

    var_corr = Variable("Correlation", is_independent=False, is_binned=False)
    var_corr.values = vals

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(var_row)
    table.add_variable(var_col)
    table.add_variable(var_corr)
    return table


def make_table_2Dscan(root_file, x_branch, y_branch, x_label, y_label, x_units, y_units, table_name, description, location, extra_filter=None, image=None):
    df = ROOT.RDataFrame("limit", root_file)

    # Best-fit
    bf_df = df.Filter("quantileExpected < -0.5 && deltaNLL==0")
    if extra_filter:
        bf_df = bf_df.Filter(extra_filter)
    bf_arrays = bf_df.AsNumpy([x_branch, y_branch])

    # Other points
    df = df.Filter("quantileExpected > -0.5 && deltaNLL > 0 && deltaNLL < 1000").Define("nll", "2*deltaNLL")
    if extra_filter:
        df = df.Filter(extra_filter)
    arrays = df.AsNumpy([x_branch, y_branch, "nll"])

    xs, ys, nll_vals = _fill_missing_grid_points(
        arrays[x_branch], arrays[y_branch], arrays["nll"]
    )

    if len(np.unique(bf_arrays[x_branch]))>1 or len(np.unique(bf_arrays[y_branch]))>1:
        print("WARNING!!! Multiple best-fit points")

    if x_branch == 'kappaH':
        print(f">>> For kappa scan, add the other best-fit")
        bf_x = np.array([bf_arrays[x_branch][0], -bf_arrays[x_branch][0]])
        bf_y = np.array([bf_arrays[y_branch][0], -bf_arrays[y_branch][0]])
    else:
        bf_x = np.array([bf_arrays[x_branch][0]])
        bf_y = np.array([bf_arrays[y_branch][0]])
    xs = np.concatenate([xs, bf_x])
    ys = np.concatenate([ys, bf_y])
    nll_vals = np.concatenate([nll_vals, np.zeros(len(bf_x))])
    print(f"  Added {len(bf_x)} best-fit point(s) with NLL=0, x: {bf_x}, y: {bf_y}")

    var_x = Variable(x_label, is_independent=True, is_binned=False, units=x_units)
    var_x.values = xs.tolist()

    var_y = Variable(y_label, is_independent=True, is_binned=False, units=y_units)
    var_y.values = ys.tolist()

    nll = Variable(r"$-2 \Delta \ln L$", is_independent=False, is_binned=False, units="")
    # IMPORTANT: 2 delta lnL
    nll.values = nll_vals.tolist()

    table = Table(table_name)
    table.description = description
    table.location = location
    if image:
        table.add_image(image)
    table.add_variable(var_x)
    table.add_variable(var_y)
    table.add_variable(nll)
    return table

submission.add_table(make_table_bdt_score(
    "input_data/ROOT/Fit_inputs.root",
    "htt_tt_1_13p6TeV",
    "Figure 2a",
    r"Genuine tau category BDT score distribution in the $\tau_h\tau_h$ channel.",
    "Data from Figure 2a",
    image="input_data/Figures/Figure_002-a.pdf",
))

submission.add_table(make_table_bdt_score(
    "input_data/ROOT/Fit_inputs.root",
    "htt_tt_2_13p6TeV",
    "Figure 2b",
    r"Mis-ID tau category BDT score distribution in the $\tau_h\tau_h$ channel.",
    "Data from Figure 2b",
    image="input_data/Figures/Figure_002-b.pdf",
))


submission.add_table(make_table_fit_inputs(
    "input_data/ROOT/Fit_inputs.root",
    "htt_tt_3_13p6TeV",
    "Bin number", "",
    "Figure 3a",
    r"$\phi_{CP}$ distribution in the $\rho\rho$ category in windows of increasing Higgs category BDT score.",
    "Data from Figure 3a",
    image="input_data/Figures/Figure_003-a.pdf",
))

submission.add_table(make_table_fit_inputs(
    "input_data/ROOT/Fit_inputs.root",
    "htt_tt_7_13p6TeV",
    "Bin number", "",
    "Figure 3b",
    r"$\phi_{CP}$ distribution in the $\pi\rho$ category in windows of increasing Higgs category BDT score.",
    "Data from Figure 3b",
    image="input_data/Figures/Figure_003-b.pdf",
))

submission.add_table(make_table_fit_inputs(
    "input_data/ROOT/Fit_inputs.root",
    "htt_tt_5_13p6TeV",
    "Bin number", "",
    "Figure 3c",
    r"$\phi_{CP}$ distribution in the $\rho a_1^{3pr}$ category in windows of increasing Higgs category BDT score.",
    "Data from Figure 3c",
    image="input_data/Figures/Figure_003-c.pdf",
))


submission.add_table(make_table_1Dscan(
    "input_data/ROOT/Run3_alpha_NLL_OBS.root",
    "outputs/Feb20_Unblinding/cmb/alpha_cmb_OBSERVED_HD",
    "Figure 4 Observed",
    r"Observed negative log-likelihood scan of $\alpha^{\mathrm{H}\tau\tau}$ for the $\sqrt{s}=13.6$ TeV data set.",
    "Data from Figure 4",
    image="input_data/Figures/Figure_004.pdf",
))

submission.add_table(make_table_1Dscan(
    "input_data/ROOT/Run3_alpha_NLL_EXP.root",
    "outputs/Feb20_Unblinding/cmb/alpha_cmb_EXPECTED_HD",
    "Figure 4 Expected",
    r"Expected negative log-likelihood scan of $\alpha^{\mathrm{H}\tau\tau}$ for the $\sqrt{s}=13.6$ TeV data set.",
    "Data from Figure 4",
    image="input_data/Figures/Figure_004.pdf",
))

submission.add_table(make_table_1Dscan(
    "input_data/ROOT/Run2Run3_alpha_NLL_OBS.root",
    "run2run3_HD_alpha_scans/alpha_cmb_OBSERVED",
    "Figure 5 Observed",
    r"Observed negative log-likelihood scan of $\alpha^{\mathrm{H}\tau\tau}$ for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets.",
    "Data from Figure 5",
    image="input_data/Figures/Figure_005.pdf",
))

submission.add_table(make_table_1Dscan(
    "input_data/ROOT/Run2Run3_alpha_NLL_EXP.root",
    "run2run3_HD_alpha_scans/alpha_cmb_EXPECTED",
    "Figure 5 Expected",
    r"Expected negative log-likelihood scan of $\alpha^{\mathrm{H}\tau\tau}$ for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets.",
    "Data from Figure 5",
    image="input_data/Figures/Figure_005.pdf",
))


submission.add_table(make_table_phiCP(
    "input_data/ROOT/Propaganda_Run3.root",
    "weighted_phiCP_10bin_categories",
    "Figure 6a",
    r"Weighted $\phi_{CP}$ distributions for the $\sqrt{s}=13.6$ TeV data set, shown for signal (SM, pseudoscalar, maximal mixing), background, and observed data.",
    "Data from Figure 6a",
    image="input_data/Figures/Figure_006-a.pdf",
))

submission.add_table(make_table_phiCP(
    "input_data/ROOT/Propaganda_Run2Run3.root",
    "weighted_phiCP_10bin_categories",
    "Figure 6b",
    r"Weighted $\phi_{CP}$ distributions for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets, shown for signal (SM, pseudoscalar, maximal mixing), background, and observed data.",
    "Data from Figure 6b",
    image="input_data/Figures/Figure_006-b.pdf",
))


submission.add_table(make_table_2Dscan(
    "input_data/ROOT/Run2Run3_mu_alpha_NLL_OBS.root",
    "alpha", "mutautau",
    r"$\alpha^{\mathrm{H}\tau\tau}$", r"$\mu$", "degrees", "",
    "Figure 7",
    r"Observed negative log-likelihood scan in the ($\alpha^{\mathrm{H}\tau\tau}$, $\mu$) plane for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets.",
    "Data from Figure 7",
    extra_filter="alpha >= -90 && alpha <= 90",
    image="input_data/Figures/Figure_007.pdf",
))

submission.add_table(make_table_2Dscan(
    "input_data/ROOT/Run2Run3_kappas_NLL_OBS.root",
    "kappaH", "kappaA",
    r"${\kappa}_\tau$", r"$\tilde{\kappa}_\tau$", "", "",
    "Figure 8",
    r"Observed negative log-likelihood scan in the (${\kappa}_\tau$, $\tilde{\kappa}_\tau$) plane for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets.",
    "Data from Figure 8",
    extra_filter="kappaH < 2.001",
    image="input_data/Figures/Figure_008.pdf",
))


submission.add_table(make_table_two_1Dscans(
    "input_data/ROOT/HLExtrap_nosyst.root", "alpha_extrap_nosysts",
    r"$-2 \Delta \ln L$ (stat. only)",
    "input_data/ROOT/HLExtrap_Run3syst.root", "alpha_extrap_run3systs",
    r"$-2 \Delta \ln L$ (stat. + syst.)",
    "Figure 9",
    r"Expected negative log-likelihood scan of $\alpha^{\mathrm{H}\tau\tau}$ for the projection to the end of the High-Luminosity LHC, with statistical uncertainties only and with systematic uncertainties kept constant with respect to the 13.6 TeV analysis.",
    "Data from Figure 9",
    image="input_data/Figures/Figure_009.pdf",
    x_range=(-15, 15),
))


submission.add_table(make_table_2Dscan(
    "input_data/ROOT/Run3_muscan.root",
    "muV", "muggH",
    r"$\mu_\mathrm{qqH}$", r"$\mu_\mathrm{ggH}$", "", "",
    "Auxillary Figure 1",
    r"Observed negative log-likelihood scan in the ($\mu_\mathrm{ggH}$, $\mu_\mathrm{qqH}$) plane for the $\sqrt{s}=13.6$ TeV data set.",
    "Data from Auxillary Figure 1",
    extra_filter="muV>=-7.25 && muV<4 && muggH>=-1 && muggH<=5",
    image="input_data/Figures/Figure_001_aux.pdf",
))


submission.add_table(make_table_correlation_matrix(
    params=[
        r"$\alpha^{\mathrm{H}\tau\tau}$",
        r"$\mu_\mathrm{qqH}$",
        r"$\mu_\mathrm{ggH}$",
    ],
    matrix=[
        [ 1.00,  0.06, -0.07],
        [ 0.06,  1.00, -0.76],
        [-0.07, -0.76,  1.00],
    ],
    table_name="Auxillary Figure 2",
    description=r"Correlation matrix for the parameters $\alpha^{\mathrm{H}\tau\tau}$, $\mu_\mathrm{qqH}$, and $\mu_\mathrm{ggH}$ from the fit to the $\sqrt{s}=13.6$ TeV data set. The best fit values are $\alpha^{\mathrm{H}\tau\tau}=(36^{+33}_{-30})^\circ$, $\mu_\mathrm{ggH} = 1.86^{+0.48}_{-0.47}$ and $\mu_\mathrm{qqH} = -1.3^{+1.1}_{-1.4}$.",
    location="Data from Auxillary Figure 2",
    image="input_data/Figures/Figure_002_aux.pdf",
))


submission.add_table(make_table_2Dscan(
    "input_data/ROOT/Run2Run3_muscan.root",
    "muV", "muggH",
    r"$\mu_\mathrm{qqH}$", r"$\mu_\mathrm{ggH}$", "", "",
    "Auxillary Figure 3",
    r"Observed negative log-likelihood scan in the ($\mu_\mathrm{ggH}$, $\mu_\mathrm{qqH}$) plane for the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets.",
    "Data from Auxillary Figure 3",
    extra_filter="muV>=-2 && muV<=3 && muggH>=-1 && muggH<=3",
    image="input_data/Figures/Figure_003_aux.pdf",
))


submission.add_table(make_table_correlation_matrix(
    params=[
        r"$\alpha^{\mathrm{H}\tau\tau}$",
        r"$\mu_\mathrm{qqH}$",
        r"$\mu_\mathrm{ggH}$",
    ],
    matrix=[
        [ 1.00,  -0.02, 0.01],
        [ -0.02,  1.00, -0.73],
        [0.01, -0.73,  1.00],
    ],
    table_name="Auxillary Figure 4",
    description=r"Correlation matrix for the parameters $\alpha^{\mathrm{H}\tau\tau}$, $\mu_\mathrm{qqH}$, and $\mu_\mathrm{ggH}$ from the fit to the $\sqrt{s}=13$ and $\sqrt{s}=13.6$ TeV data sets. The best fit values are $\alpha^{\mathrm{H}\tau\tau}=(7 \pm 16)^\circ$, $\mu_\mathrm{ggH} = 0.93^{+0.29}_{-0.25}$ and $\mu_\mathrm{qqH} = 0.79^{+0.50}_{-0.47}$.",
    location="Data from Auxillary Figure 4",
    image="input_data/Figures/Figure_004_aux.pdf",
))



submission.create_files("output")
print("Done — files written to output/")
