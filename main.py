import streamlit as st
import math
import numpy as np
from scipy.optimize import fsolve, brentq
import plotly.graph_objects as go
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Compressible Flow Calculator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


class CompressibleFlowCalculator:
    def __init__(self):
        self.gamma = 1.4  # Default specific heat ratio for air

    def set_gamma(self, gamma):
        """Set specific heat ratio"""
        self.gamma = float(gamma)

    # ===============================
    # ISENTROPIC FLOW RELATIONS
    # ===============================
    def mach_to_pressure_ratio(self, M):
        """Calculate pressure ratio P/P0 from Mach number"""
        return (1 + (self.gamma - 1) / 2 * M ** 2) ** (-self.gamma / (self.gamma - 1))

    def mach_to_temperature_ratio(self, M):
        """Calculate temperature ratio T/T0 from Mach number"""
        return (1 + (self.gamma - 1) / 2 * M ** 2) ** (-1)

    def mach_to_density_ratio(self, M):
        """Calculate density ratio rho/rho0 from Mach number"""
        return (1 + (self.gamma - 1) / 2 * M ** 2) ** (-1 / (self.gamma - 1))

    def mach_to_area_ratio(self, M):
        """Calculate area ratio A/A* from Mach number"""
        if M <= 0:
            return np.nan
        term1 = 1.0 / M
        term2 = ((2.0 / (self.gamma + 1.0)) * (1.0 + (self.gamma - 1.0) / 2.0 * M ** 2)) ** (
                (self.gamma + 1.0) / (2.0 * (self.gamma - 1.0))
        )
        return term1 * term2

    def calculate_throat_area(self, A, M):
        """Calculate throat area A* from given area A and Mach number M"""
        area_ratio = self.mach_to_area_ratio(M)
        if area_ratio is None or np.isnan(area_ratio) or area_ratio <= 0:
            return None
        return A / area_ratio

    def calculate_area_from_throat(self, A_star, M):
        """Calculate area A from throat area A* and Mach number M"""
        area_ratio = self.mach_to_area_ratio(M)
        if area_ratio is None or np.isnan(area_ratio):
            return None
        return A_star * area_ratio

    def area_ratio_to_mach_subsonic(self, area_ratio):
        """Calculate subsonic Mach number from area ratio A/A* using brentq"""
        if area_ratio <= 0:
            return None

        def f(M):
            return self.mach_to_area_ratio(M) - area_ratio

        try:
            # subsonic root is between a small positive number and 1
            return brentq(f, 1e-6, 1.0 - 1e-6, maxiter=200)
        except Exception:
            return None

    def area_ratio_to_mach_supersonic(self, area_ratio):
        """Calculate supersonic Mach number from area ratio A/A* using brentq"""
        if area_ratio <= 0:
            return None

        def f(M):
            return self.mach_to_area_ratio(M) - area_ratio

        try:
            # supersonic root between just above 1 and a large Mach (e.g. 100)
            return brentq(f, 1.0 + 1e-6, 200.0, maxiter=200)
        except Exception:
            return None

    def pressure_ratio_to_mach(self, pressure_ratio):
        """Calculate Mach number from pressure ratio P/P0"""
        if pressure_ratio <= 0 or pressure_ratio > 1.0:
            return None
        term = (pressure_ratio ** (-(self.gamma - 1.0) / self.gamma) - 1.0) * 2.0 / (self.gamma - 1.0)
        if term <= 0:
            return None
        return math.sqrt(term)

    def temperature_ratio_to_mach(self, temp_ratio):
        """Calculate Mach number from temperature ratio T/T0"""
        if temp_ratio <= 0 or temp_ratio > 1.0:
            return None
        term = (1.0 / temp_ratio - 1.0) * 2.0 / (self.gamma - 1.0)
        if term <= 0:
            return None
        return math.sqrt(term)

    def density_ratio_to_mach(self, density_ratio):
        """Calculate Mach number from density ratio rho/rho0"""
        if density_ratio <= 0 or density_ratio > 1.0:
            return None
        term = (density_ratio ** (-(self.gamma - 1.0)) - 1.0) * 2.0 / (self.gamma - 1.0)
        if term <= 0:
            return None
        return math.sqrt(term)

    # ===============================
    # PRANDTL-MEYER EXPANSION
    # ===============================
    def prandtl_meyer_angle(self, M):
        """Calculate Prandtl-Meyer angle (degrees) from Mach number"""
        if M <= 1.0:
            return 0.0
        g = self.gamma
        term1 = math.sqrt((g + 1.0) / (g - 1.0))
        term2 = math.atan(math.sqrt((g - 1.0) / (g + 1.0) * (M ** 2 - 1.0)))
        term3 = math.atan(math.sqrt(M ** 2 - 1.0))
        nu = term1 * term2 - term3
        return math.degrees(nu)

    def mach_from_prandtl_meyer(self, nu_degrees):
        """Calculate Mach number from Prandtl-Meyer angle (degrees) using brentq inversion"""
        if nu_degrees <= 0:
            return None
        nu_target = nu_degrees  # keep in degrees because prandtl_meyer_angle returns degrees

        # maximum possible nu for practical large Mach (approximate)
        max_nu = self.prandtl_meyer_angle(1e4)
        if nu_target >= max_nu:
            return None

        def f(M):
            return self.prandtl_meyer_angle(M) - nu_target

        try:
            # monotonic in M for M>1 -> we can bracket between (1+eps) and some large M
            return brentq(f, 1.0 + 1e-6, 1e4, maxiter=200)
        except Exception:
            return None

    # ===============================
    # NORMAL SHOCK RELATIONS
    # ===============================
    def normal_shock_pressure_ratio(self, M1):
        """Calculate pressure ratio P2/P1 across normal shock"""
        g = self.gamma
        return 1.0 + (2.0 * g / (g + 1.0)) * (M1 ** 2 - 1.0)

    def normal_shock_density_ratio(self, M1):
        """Calculate density ratio rho2/rho1 across normal shock"""
        g = self.gamma
        return ((g + 1.0) * M1 ** 2) / ((g - 1.0) * M1 ** 2 + 2.0)

    def normal_shock_temperature_ratio(self, M1):
        """Calculate temperature ratio T2/T1 across normal shock (use p/rho to be robust)"""
        p_ratio = self.normal_shock_pressure_ratio(M1)
        rho_ratio = self.normal_shock_density_ratio(M1)
        if rho_ratio == 0:
            return None
        return p_ratio / rho_ratio

    def normal_shock_mach2(self, M1):
        """Calculate downstream Mach number M2 from upstream M1"""
        g = self.gamma
        # standard algebraic form (stable)
        num = 1.0 + (g - 1.0) / 2.0 * M1 ** 2
        den = g * M1 ** 2 - (g - 1.0) / 2.0
        if den <= 0:
            return None
        return math.sqrt(num / den)

    def normal_shock_stagnation_pressure_ratio(self, M1):
        """Calculate stagnation pressure ratio P02/P01 across normal shock"""
        g = self.gamma
        term1 = ((g + 1.0) * M1 ** 2 / (2.0 + (g - 1.0) * M1 ** 2)) ** (g / (g - 1.0))
        term2 = ((g + 1.0) / (2.0 * g * M1 ** 2 - (g - 1.0))) ** (1.0 / (g - 1.0))
        return term1 * term2

    # ===============================
    # OBLIQUE SHOCK RELATIONS
    # ===============================
    def oblique_shock_beta_from_theta_M(self, theta_degrees, M1, strong=False):
        """Calculate shock angle beta (degrees) from deflection angle theta (deg) and M1.
           Returns None on failure. Uses fsolve but enforces physically-valid range.
        """
        theta_rad = math.radians(theta_degrees)
        g = self.gamma

        # physical lower bound for beta is asin(1/M1) (Mach angle), up to pi/2
        try:
            lower = math.asin(min(max(1.0 / M1, -1.0), 1.0)) + 1e-9
        except Exception:
            return None
        upper = math.pi / 2.0 - 1e-9

        def residual(beta_rad):
            # avoid tangent singularities
            if abs(math.tan(beta_rad)) < 1e-12:
                return 1e6
            num = M1 ** 2 * math.sin(beta_rad) ** 2 - 1.0
            den = M1 ** 2 * (g + math.cos(2.0 * beta_rad)) + 2.0
            rhs = 2.0 / math.tan(beta_rad) * (num / den)
            return rhs - math.tan(theta_rad)

        # initial guesses: weak shock near lower bound, strong shock closer to upper
        initial = lower + (upper - lower) * (0.25 if not strong else 0.75)
        try:
            beta_solution = fsolve(residual, initial, xtol=1e-9, maxfev=200)[0]
            # ensure real and within bounds
            if not np.isreal(beta_solution):
                return None
            if beta_solution <= lower - 1e-8 or beta_solution >= upper + 1e-8:
                return None
            return math.degrees(float(abs(beta_solution)))
        except Exception:
            return None

    def oblique_shock_theta_from_beta_M(self, beta_degrees, M1):
        """Calculate deflection angle theta (degrees) from shock angle beta (deg) and M1"""
        beta_rad = math.radians(beta_degrees)
        M1n = M1 * math.sin(beta_rad)  # Normal component of M1

        if M1n <= 1:
            return None  # No shock possible (should be supersonic normal component)

        g = self.gamma
        num = 2.0 * (M1 ** 2 * math.sin(beta_rad) ** 2 - 1.0)
        den = M1 ** 2 * (g + math.cos(2.0 * beta_rad)) + 2.0
        tan_theta = num / den / math.tan(beta_rad)
        # tan_theta can be negative if invalid; check
        try:
            theta = math.atan(tan_theta)
            return math.degrees(theta)
        except Exception:
            return None

    def oblique_shock_pressure_ratio(self, beta_degrees, M1):
        """Calculate pressure ratio P2/P1 across oblique shock"""
        beta_rad = math.radians(beta_degrees)
        M1n = M1 * math.sin(beta_rad)
        return self.normal_shock_pressure_ratio(M1n)

    def oblique_shock_density_ratio(self, beta_degrees, M1):
        """Calculate density ratio rho2/rho1 across oblique shock"""
        beta_rad = math.radians(beta_degrees)
        M1n = M1 * math.sin(beta_rad)
        return self.normal_shock_density_ratio(M1n)

    def oblique_shock_temperature_ratio(self, beta_degrees, M1):
        """Calculate temperature ratio T2/T1 across oblique shock"""
        beta_rad = math.radians(beta_degrees)
        M1n = M1 * math.sin(beta_rad)
        return self.normal_shock_temperature_ratio(M1n)

    def oblique_shock_mach2(self, beta_degrees, theta_degrees, M1):
        """Calculate downstream Mach number M2 (full Mach number after oblique shock)"""
        beta_rad = math.radians(beta_degrees)
        theta_rad = math.radians(theta_degrees)
        M1n = M1 * math.sin(beta_rad)
        M2n = self.normal_shock_mach2(M1n)
        if M2n is None:
            return None
        denom = math.sin(beta_rad - theta_rad)
        if abs(denom) < 1e-12:
            return None
        return M2n / denom


# Initialize calculator
@st.cache_resource
def get_calculator():
    return CompressibleFlowCalculator()


calc = get_calculator()

# App Title
st.title(" Compressible Flow Calculator")
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
gamma = st.sidebar.slider("Specific Heat Ratio (γ)", 1.1, 2.0, 1.4, 0.01, key="sidebar_gamma")
calc.set_gamma(gamma)

st.sidebar.markdown(f"**Current γ = {gamma}**")
st.sidebar.markdown("Common values:")
st.sidebar.markdown("- Air: γ = 1.4")
st.sidebar.markdown("- Helium: γ = 1.67")
st.sidebar.markdown("- CO₂: γ = 1.3")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Isentropic Flow",
    " A* Calculator",
    " Prandtl-Meyer",
    " Normal Shock",
    " Oblique Shock",

])

# Tab 1: Isentropic Flow Relations
with tab1:
    st.header("Isentropic Flow Relations")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        input_type = st.selectbox(
            "Select Input Parameter:",
            ["Mach Number", "Pressure Ratio (P/P₀)", "Temperature Ratio (T/T₀)",
             "Density Ratio (ρ/ρ₀)", "Area Ratio (A/A*)"],
            key="tab1_input_type"
        )

        M = None
        p_ratio = T_ratio = rho_ratio = A_ratio = None

        if input_type == "Mach Number":
            M = st.number_input("Mach Number", 0.01, 100.0, 2.0, 0.01, key="tab1_M")
            p_ratio = calc.mach_to_pressure_ratio(M)
            T_ratio = calc.mach_to_temperature_ratio(M)
            rho_ratio = calc.mach_to_density_ratio(M)
            A_ratio = calc.mach_to_area_ratio(M)

        elif input_type == "Pressure Ratio (P/P₀)":
            p_ratio = st.number_input("Pressure Ratio (P/P₀)", 1e-6, 1.0, 0.5, 0.01, key="tab1_p_ratio")
            M = calc.pressure_ratio_to_mach(p_ratio)
            if M is not None:
                T_ratio = calc.mach_to_temperature_ratio(M)
                rho_ratio = calc.mach_to_density_ratio(M)
                A_ratio = calc.mach_to_area_ratio(M)
            else:
                st.error("Invalid pressure ratio - no physical Mach number found.")

        elif input_type == "Temperature Ratio (T/T₀)":
            T_ratio = st.number_input("Temperature Ratio (T/T₀)", 1e-6, 1.0, 0.8, 0.01, key="tab1_T_ratio")
            M = calc.temperature_ratio_to_mach(T_ratio)
            if M is not None:
                p_ratio = calc.mach_to_pressure_ratio(M)
                rho_ratio = calc.mach_to_density_ratio(M)
                A_ratio = calc.mach_to_area_ratio(M)
            else:
                st.error("Invalid temperature ratio - no physical Mach number found.")

        elif input_type == "Density Ratio (ρ/ρ₀)":
            rho_ratio = st.number_input("Density Ratio (ρ/ρ₀)", 1e-6, 1.0, 0.6, 0.01, key="tab1_rho_ratio")
            M = calc.density_ratio_to_mach(rho_ratio)
            if M is not None:
                p_ratio = calc.mach_to_pressure_ratio(M)
                T_ratio = calc.mach_to_temperature_ratio(M)
                A_ratio = calc.mach_to_area_ratio(M)
            else:
                st.error("Invalid density ratio - no physical Mach number found.")

        elif input_type == "Area Ratio (A/A*)":
            A_ratio = st.number_input("Area Ratio (A/A*)", 1.0, 500.0, 2.0, 0.1, key="tab1_A_ratio")

            # Calculate both subsonic and supersonic solutions
            M_sub = calc.area_ratio_to_mach_subsonic(A_ratio)
            M_sup = calc.area_ratio_to_mach_supersonic(A_ratio)

            if M_sub is None and M_sup is None:
                st.error("Could not find subsonic or supersonic Mach for this A/A*.")
            else:
                if M_sub is not None and M_sup is not None:
                    flow_regime = st.radio("Select Flow Regime:", ["Subsonic", "Supersonic"], key="tab1_regime")
                    M = M_sub if flow_regime == "Subsonic" else M_sup
                elif M_sub is not None:
                    M = M_sub
                else:
                    M = M_sup

                if M is not None:
                    p_ratio = calc.mach_to_pressure_ratio(M)
                    T_ratio = calc.mach_to_temperature_ratio(M)
                    rho_ratio = calc.mach_to_density_ratio(M)

    with col2:
        st.subheader("Results")

        if M is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mach Number", f"{M:.6f}")
                if p_ratio is not None:
                    st.metric("Pressure Ratio (P/P₀)", f"{p_ratio:.6f}")
                if T_ratio is not None:
                    st.metric("Temperature Ratio (T/T₀)", f"{T_ratio:.6f}")

            with col_b:
                if rho_ratio is not None:
                    st.metric("Density Ratio (ρ/ρ₀)", f"{rho_ratio:.6f}")
                if A_ratio is not None:
                    st.metric("Area Ratio (A/A*)", f"{A_ratio:.6f}")

                # Flow regime indicator
                if M < 1:
                    st.success("🔵 Subsonic Flow")
                elif abs(M - 1.0) < 1e-6:
                    st.info("🟡 Sonic Flow")
                else:
                    st.warning("🔴 Supersonic Flow")
        else:
            st.info("Enter inputs to compute isentropic flow properties.")

# Tab 2: A* (Throat Area) Calculator
with tab2:
    st.header("A* (Throat Area) Calculator")

    st.markdown("""
    Calculate the throat area (A*) required for a given flow condition, or calculate 
    the area at any location given the throat area and local Mach number.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        calc_mode = st.selectbox(
            "Calculation Mode:",
            ["Calculate A* (given A and M)", "Calculate A (given A* and M)"],
            key="tab2_calc_mode"
        )

        if calc_mode == "Calculate A* (given A and M)":
            A_input = st.number_input("Area A (m² or any unit)", 0.001, 1000.0, 1.0, 0.001,
                                      key="tab2_A_input", format="%.6f")
            M_input = st.number_input("Local Mach Number", 0.01, 100.0, 2.0, 0.01,
                                      key="tab2_M_input")

            # Calculate A*
            A_star = calc.calculate_throat_area(A_input, M_input)
            area_ratio_display = calc.mach_to_area_ratio(M_input)

        else:  # Calculate A (given A* and M)
            A_star_input = st.number_input("Throat Area A* (m² or any unit)", 0.001, 1000.0, 0.5, 0.001,
                                           key="tab2_Astar_input", format="%.6f")
            M_input = st.number_input("Local Mach Number", 0.01, 100.0, 2.0, 0.01,
                                      key="tab2_M_input2")

            # Calculate A
            A_calculated = calc.calculate_area_from_throat(A_star_input, M_input)
            area_ratio_display = calc.mach_to_area_ratio(M_input)
            A_star = A_star_input

    with col2:
        st.subheader("Results")

        if calc_mode == "Calculate A* (given A and M)":
            if A_star is not None and not np.isnan(A_star):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Given Area (A)", f"{A_input:.6f}")
                    st.metric("Mach Number (M)", f"{M_input:.6f}")

                with col_b:
                    st.metric("Throat Area (A*)", f"{A_star:.6f}",
                              help="This is the minimum (sonic) area required")
                    st.metric("Area Ratio (A/A*)", f"{area_ratio_display:.6f}")

                # Flow regime
                if M_input < 1:
                    st.success("🔵 Subsonic Flow - Area is converging towards throat")
                elif abs(M_input - 1.0) < 1e-6:
                    st.info("🟡 Sonic Flow - At throat location (A = A*)")
                else:
                    st.warning("🔴 Supersonic Flow - Area is diverging from throat")

                # Additional info
                st.markdown("---")
                st.markdown("**Physical Interpretation:**")
                if M_input < 1:
                    st.write(
                        f"• To accelerate flow from subsonic to sonic, reduce area from {A_input:.4f} to {A_star:.4f}")
                    st.write(f"• Area reduction ratio: {(A_input / A_star):.3f}:1")
                elif M_input > 1:
                    st.write(f"• Flow accelerated from sonic at throat (A* = {A_star:.4f}) to supersonic")
                    st.write(f"• Area expansion ratio: {(A_input / A_star):.3f}:1")
                else:
                    st.write(f"• Flow is at sonic condition (throat location)")

            else:
                st.error("Could not calculate A* for the given conditions.")

        else:  # Calculate A mode
            if A_calculated is not None and not np.isnan(A_calculated):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Throat Area (A*)", f"{A_star_input:.6f}")
                    st.metric("Mach Number (M)", f"{M_input:.6f}")

                with col_b:
                    st.metric("Calculated Area (A)", f"{A_calculated:.6f}",
                              help="Area at the location with specified Mach number")
                    st.metric("Area Ratio (A/A*)", f"{area_ratio_display:.6f}")

                # Flow regime
                if M_input < 1:
                    st.success("🔵 Subsonic Flow")
                elif abs(M_input - 1.0) < 1e-6:
                    st.info("🟡 Sonic Flow")
                else:
                    st.warning("🔴 Supersonic Flow")

                # Additional info
                st.markdown("---")
                st.markdown("**Nozzle Design Information:**")
                st.write(f"• Throat area: {A_star_input:.4f}")
                st.write(f"• Area at M = {M_input:.2f}: {A_calculated:.4f}")
                st.write(f"• Required area change: {abs(A_calculated - A_star_input):.4f}")
                if M_input < 1:
                    st.write(
                        f"• Converging section: area decreases by {((1 - A_calculated / A_star_input) * 100):.1f}% to reach throat")
                elif M_input > 1:
                    st.write(
                        f"• Diverging section: area increases by {((A_calculated / A_star_input - 1) * 100):.1f}% from throat")
            else:
                st.error("Could not calculate area for the given conditions.")

# Tab 3: Prandtl-Meyer Expansion
with tab3:
    st.header("Prandtl-Meyer Expansion")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        pm_input = st.selectbox(
            "Select Input:",
            ["Mach Number", "Prandtl-Meyer Angle (ν)"],
            key="tab2_pm_input"
        )

        if pm_input == "Mach Number":
            M_pm = st.number_input("Mach Number (M)", 1.0001, 1e4, 2.0, 0.01, key="tab2_M_pm")
            nu = calc.prandtl_meyer_angle(M_pm)
        else:
            nu = st.number_input("Prandtl-Meyer Angle (degrees)", 0.0, 180.0, 20.0, 0.1, key="tab2_nu")
            M_pm = calc.mach_from_prandtl_meyer(nu)

    with col2:
        st.subheader("Results")

        if M_pm is not None and M_pm > 1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mach Number", f"{M_pm:.6f}")
            with col_b:
                st.metric("Prandtl-Meyer Angle (ν)", f"{nu:.6f}°")

            # Maximum possible angle (practical large Mach)
            max_nu = calc.prandtl_meyer_angle(1e4)
            st.info(f"Maximum possible ν for γ={gamma}: {max_nu:.2f}°")
        else:
            st.error("Prandtl-Meyer expansion valid only for supersonic flow (M > 1).")

# Tab 4: Normal Shock Relations
with tab4:
    st.header("Normal Shock Relations")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")
        M1_ns = st.number_input("Upstream Mach Number (M₁)", 1.01, 100.0, 2.0, 0.01, key="tab3_M1")

    with col2:
        st.subheader("Results")

        if M1_ns > 1:
            M2_ns = calc.normal_shock_mach2(M1_ns)
            p_ratio_ns = calc.normal_shock_pressure_ratio(M1_ns)
            rho_ratio_ns = calc.normal_shock_density_ratio(M1_ns)
            T_ratio_ns = calc.normal_shock_temperature_ratio(M1_ns)
            p0_ratio_ns = calc.normal_shock_stagnation_pressure_ratio(M1_ns)

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Upstream Mach (M₁)", f"{M1_ns:.6f}")
                st.metric("Downstream Mach (M₂)", f"{M2_ns:.6f}" if M2_ns is not None else "N/A")
                st.metric("Pressure Ratio (P₂/P₁)", f"{p_ratio_ns:.6f}")

            with col_b:
                st.metric("Density Ratio (ρ₂/ρ₁)", f"{rho_ratio_ns:.6f}")
                st.metric("Temperature Ratio (T₂/T₁)", f"{T_ratio_ns:.6f}" if T_ratio_ns is not None else "N/A")
                st.metric("Stagnation Pressure Ratio (P₀₂/P₀₁)", f"{p0_ratio_ns:.6f}")
        else:
            st.error("Normal shock requires supersonic upstream flow (M₁ > 1)")

# Tab 5: Oblique Shock Relations
with tab5:
    st.header("Oblique Shock Relations")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        oblique_mode = st.selectbox(
            "Calculation Mode:",
            ["Given θ and M₁ (find β)", "Given β and M₁ (find θ)"],
            key="tab4_mode"
        )

        M1_os = st.number_input("Upstream Mach Number (M₁)", 1.01, 100.0, 2.0, 0.01, key="tab4_M1")

        if oblique_mode == "Given θ and M₁ (find β)":
            theta_os = st.number_input("Deflection Angle θ (degrees)", 0.0, 89.0, 10.0, 0.1, key="tab4_theta")
            show_both = st.checkbox("Show both weak and strong shock solutions", key="tab4_showboth")
        else:
            beta_os = st.number_input("Shock Angle β (degrees)", 0.0, 90.0, 30.0, 0.1, key="tab4_beta")

    with col2:
        st.subheader("Results")

        if M1_os > 1:
            if oblique_mode == "Given θ and M₁ (find β)":
                beta_weak = calc.oblique_shock_beta_from_theta_M(theta_os, M1_os, strong=False)
                beta_strong = calc.oblique_shock_beta_from_theta_M(theta_os, M1_os, strong=True)

                solutions = []
                if beta_weak:
                    solutions.append(("Weak", beta_weak))
                if beta_strong and show_both:
                    solutions.append(("Strong", beta_strong))

                for shock_type, beta in solutions:
                    st.markdown(f"**{shock_type} Shock Solution:**")
                    M2_os = calc.oblique_shock_mach2(beta, theta_os, M1_os)
                    p_ratio_os = calc.oblique_shock_pressure_ratio(beta, M1_os)
                    rho_ratio_os = calc.oblique_shock_density_ratio(beta, M1_os)
                    T_ratio_os = calc.oblique_shock_temperature_ratio(beta, M1_os)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Shock Angle (β)", f"{beta:.3f}°")
                        st.metric("Downstream Mach (M₂)", f"{M2_os:.6f}" if M2_os is not None else "N/A")
                    with col_b:
                        st.metric("Pressure Ratio (P₂/P₁)", f"{p_ratio_os:.6f}")
                        st.metric("Temperature Ratio (T₂/T₁)", f"{T_ratio_os:.6f}" if T_ratio_os is not None else "N/A")

                    st.markdown("---")

                if not solutions:
                    st.error("No physically-valid shock solution exists for these conditions!")
            else:  # Given β and M₁
                theta_calculated = calc.oblique_shock_theta_from_beta_M(beta_os, M1_os)

                if theta_calculated is not None and theta_calculated > 0:
                    M2_os = calc.oblique_shock_mach2(beta_os, theta_calculated, M1_os)
                    p_ratio_os = calc.oblique_shock_pressure_ratio(beta_os, M1_os)
                    rho_ratio_os = calc.oblique_shock_density_ratio(beta_os, M1_os)
                    T_ratio_os = calc.oblique_shock_temperature_ratio(beta_os, M1_os)

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric("Deflection Angle (θ)", f"{theta_calculated:.3f}°")
                        st.metric("Downstream Mach (M₂)", f"{M2_os:.6f}" if M2_os is not None else "N/A")

                    with col_b:
                        st.metric("Pressure Ratio (P₂/P₁)", f"{p_ratio_os:.6f}")
                        st.metric("Temperature Ratio (T₂/T₁)", f"{T_ratio_os:.6f}" if T_ratio_os is not None else "N/A")
                else:
                    st.error("Invalid shock angle for given conditions!")

        else:
            st.error("Oblique shock requires supersonic upstream flow (M₁ > 1)")

st.markdown("---")
