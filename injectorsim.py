class injector:
    def set_fuel_CdA(self, CdA):
        __doc__ = """
            Sets the fuel injector CdA.

            Parameters
            ----------
            CdA : float
                Product of fuel discharge coefficient and area.
            """
        self.fuel_CdA = CdA
        self.calc_film()

    def set_ox_CdA(self, CdA):
        __doc__ = """
            Sets the oxidizer injector CdA.

            Parameters
            ----------
            CdA : float
                Product of oxidizer discharge coefficient and area.
            """
        self.ox_CdA = CdA

    def size_fuel_anulus(self, Cd, ID, OD, n = 1):
        __doc__ = """
            Sizes fuel injector for a number of identical annular holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the fuel annulus.
            ID : float
                Inner diameter of the annulus in millimeters.
            OD : float
                Outer diameter of the annulus in millimeters.
            n : int, optional
                Number of annular holes (default 1).
            """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25 * np.pi * (OD**2 - ID**2) * n
        self.fuel_CdA = self.fuel_A * Cd
        self.calc_film()
    
    def size_ox_anulus(self, Cd, ID, OD, n = 1):
        __doc__ = """
            Sizes oxidizer injector for a number of identical annular holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the oxidizer annulus.
            ID : float
                Inner diameter of the annulus in meters.
            OD : float
                Outer diameter of the annulus in meters.
            n : int, optional
                Number of annular holes (default 1).
            """
        self.ox_Cd = Cd
        self.ox_A = 0.25 * np.pi * (OD**2 - ID**2) * n
        self.ox_CdA = self.ox_A * Cd

    def size_fuel_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the fuel injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the fuel holes.
            d : float
                Hole diameter in metres.
            n : int, optional
                Number of fuel holes (default 1).
            """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25 * np.pi * (d**2) * n
        self.fuel_CdA = self.fuel_A * Cd
        self.calc_film()
    
    def size_film_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the film cooling injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the film cooling holes.
            d : float
                Hole diameter in metres.
            n : int, optional
                Number of film cooling holes (default 1).
            """
        self.film_Cd = Cd
        self.film_A = 0.25 * np.pi * (d**2) * n
        self.film_CdA = self.film_A * Cd
        self.calc_film()

    def size_ox_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the oxidizer injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the oxidizer holes.
            d : float
                Hole diameter in metres.
            n : int, optional
                Number of oxidizer holes (default 1).
            """
        self.ox_Cd = Cd
        self.ox_A = 0.25 * np.pi * (d**2) * n
        self.ox_CdA = self.ox_A * Cd

    def spi_fuel_mdot(self, dp, fuel_rho):
        __doc__ = """
            Calculates the fuel mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (Pa)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Fuel mass flow rate (kg/s)
            """
        return self.fuel_CdA * np.sqrt(2 * dp * fuel_rho)

    def spi_ox_mdot(self, dp, ox_rho):
        __doc__ = """
            Calculates the oxidiser mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (bar)
            ox_rho : float
                Density of the oxidiser (kg/m^3)

            Returns
            -------
            float
                Oxidiser mass flow rate (kg/s)
            """
        return self.ox_CdA * np.sqrt(2e5 * dp * ox_rho)

    def spi_fuel_core_dp(self, mdot, fuel_rho):
        __doc__ = """
            Calculates the pressure differential across the fuel injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Fuel mass flow rate (kg/s)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.fuel_CdA)**2) / (2e5 * fuel_rho)

    def spi_fuel_total_dp(self, mdot, fuel_rho):
        __doc__ = """
            Calculates the pressure differential across the total fuel injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Total fuel mass flow rate (kg/s)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.fuel_CdA)**2) / (2 * fuel_rho)

    def spi_film_dp(self, mdot, film_rho):
        __doc__ = """
            Calculates the pressure differential across the film cooling injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Film cooling mass flow rate (kg/s)
            film_rho : float
                Density of the film coolant (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.film_CdA)**2) / (2e5 * film_rho)
   
    def spi_ox_dp(self, mdot, ox_rho):
        __doc__ = """
            Calculates the pressure differential across the oxidizer injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Oxidizer mass flow rate (kg/s)
            ox_rho : float
                Density of the oxidizer (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.ox_CdA)**2) / (2 * ox_rho)
    
    def ox_flow_setup(self, ox_class, ox_downstream_p, ox_upstream_p, ox_upstream_T, ox_vp):
        self.ox_downstream_p = ox_downstream_p
        self.ox_upstream_p = ox_upstream_p
        self.ox_upstream_T = ox_upstream_T
        self.ox_vp = ox_vp

        self.ox_up = Fluid(ox_class)
        self.ox_down = Fluid(ox_class)

        if ox_upstream_p == None:
            self.ox_saturated = True
        else:
            self.ox_saturated = False
        
        if self.ox_upstream_T == None and self.ox_vp == None:
            raise ValueError("Either upstream_T or vp must be provided.")
        elif self.ox_upstream_T is not None and self.ox_vp is not None:
            raise ValueError("Both upstream_T and vp cannot be provided.")
        elif self.ox_upstream_T is not None:
            if self.ox_saturated:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.quality(0))
                self.ox_vp = self.ox_up.pressure
                self.ox_upstream_p = self.ox_vp
            else:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.pressure(self.ox_upstream_p))
        elif self.ox_vp is not None:
            self.ox_up.update(Input.pressure(self.ox_vp), Input.quality(0))
            self.ox_upstream_T = self.ox_up.temperature
            if self.ox_saturated:
                self.ox_upstream_p = self.ox_vp + 10
            else:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.pressure(self.ox_upstream_p))
        
        self.ox_down.update(Input.pressure(self.ox_downstream_p), Input.entropy(self.ox_up.entropy))

    def hem_ox_mdot(self, ox_class: Fluid, downstream_p, upstream_p=None, upstream_T = None, vp = None):
        __doc__ = """
            Calculates the oxidizer mass flow rate using the HEM model.
            Used for fluids that can exhibit two phase flow.

            Parameters
            ----------
            ox_class : object
                pyfluids object for oxidizer
            downstream_p : float
                Downstream pressure (bar)
            upstream_p : float, optional
                Upstream pressure (bar), defaults to None
            upstream_T : float, optional
                Upstream temperature (°C), defaults to None
            vp : float, optional
                Vapor pressure (bar), defaults to None

            Returns
            -------
            float
                Oxidizer mass flow rate (kg/s)
            """

        self.ox_flow_setup(ox_class, downstream_p, upstream_p, upstream_T, vp)

        def HEMfunc(up, down, downstream_p):
            down.update(Input.pressure(downstream_p), Input.entropy(up.entropy))
            return self.ox_CdA * down.density * np.sqrt(2 * (up.enthalpy - down.enthalpy))

        sol = sp.optimize.minimize_scalar(lambda x: -HEMfunc(self.ox_up, self.ox_down, x), bounds=[0,self.ox_upstream_p], method='bounded')

        self.ox_choked_p = sol.x
        choked_mdot = -sol.fun

        if (self.ox_choked_p > self.ox_downstream_p):
            mdot = choked_mdot
        else:
            mdot = HEMfunc(self.ox_up, self.ox_down, self.ox_downstream_p)

        mdot = 0 if np.isnan(mdot) else mdot

        return mdot

    def nhne_ox_mdot(self, ox_class, downstream_p, upstream_p=None, upstream_T = None, vp = None):
        __doc__ = """
            Calculates the oxidizer mass flow rate using the NHNE model.
            Used for fluids that can exhibit two phase flow.

            Parameters
            ----------
            ox_class : object
                pyfluids object for oxidizer
            downstream_p : float
                Downstream pressure (bar)
            upstream_p : float, optional
                Upstream pressure (bar), defaults to None
            upstream_T : float, optional
                Upstream temperature (°C), defaults to None
            vp : float, optional
                Vapor pressure (bar), defaults to None

            Returns
            -------
            float
                Oxidizer mass flow rate (kg/s)
            """

        HEM_mdot = self.hem_ox_mdot(ox_class, downstream_p, upstream_p, upstream_T, vp)
        SPI_mdot = self.spi_ox_mdot((self.ox_upstream_p - self.ox_downstream_p), self.ox_up.density)

        k = np.sqrt((self.ox_upstream_p - self.ox_downstream_p) / (self.ox_vp - self.ox_downstream_p)) if self.ox_downstream_p < self.ox_vp else 1

        mdot = (SPI_mdot* k / (1 + k)) + (HEM_mdot / (1 + k))

        return mdot

    def calc_start_mdot(self, fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, ox_temp=15, fuel_gas_class=None, fuel_temp=15):
        __doc__ = """
            Calculates the starting mdots for the injector (venting to atm).
            Disregards film cooling.
            ----------
            fuel_inj_p : float
                Fuel injector pressure (bar)
            ox_inj_p : float
                Oxidizer injector pressure (bar)
            fuel_rho : float, optional
                Fuel density in kg/m³, defaults to 786
            ox_rho : float, optional
                Oxidizer density in kg/m³, defaults to 860 (used only if oxclass is None)
            oxclass : object, optional
                pyfluids object for oxidizer
                If provided, compressible flow calculations will be used for oxidizer
            ox_temp : float, optional
                Oxidizer temperature in °C, defaults to 15 (used only if oxclass is provided)
            Returns
            -------
            None
                Results are printed directly:
                - Total mass flow rate (g/s)
                - Oxidizer mass flow rate (g/s) and whether flow is choked
                - Fuel mass flow rate (g/s)
                - Oxidizer to fuel ratio (OF)
            """
        if ox_gas_class != None:
            (ox_mdot_start, _, ox_chokedstate) = spc_mdot(self.ox_CdA, ox_gas_class, degC_to_K(ox_temp), ox_inj_p, 101325)
        else:
            ox_mdot_start = self.ox_CdA * np.sqrt(2*(ox_inj_p-101325) * ox_rho)

        if fuel_gas_class != None:
            (fuel_mdot_start, _, fuel_chokedstate) = spc_mdot(self.fuel_CdA, fuel_gas_class, degC_to_K(fuel_temp), fuel_inj_p, 101325)
        else:
            fuel_mdot_start = self.fuel_CdA * np.sqrt(2 * (fuel_inj_p-101325) * fuel_rho)

        print(f'Total Start mdot: {(ox_mdot_start+fuel_mdot_start)*1e3:.4f} g/s')
        if ox_gas_class != None:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.4f} g/s, choked: {ox_chokedstate}')
        else:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.4f} g/s')
        if fuel_gas_class != None:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.4f} g/s, choked: {fuel_chokedstate}')
        else:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.4f} g/s')
        print(f'Start OF: {ox_mdot_start/fuel_mdot_start:.3f}')