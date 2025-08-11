--
-- PostgreSQL database dump
--

-- Dumped from database version 14.6
-- Dumped by pg_dump version 14.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: gaia_source; Type: TABLE; Schema: public; Owner: tglctester
--

CREATE TABLE public.gaia_source (
    solution_id bigint NOT NULL,
    designation character varying(32) NOT NULL,
    source_id bigint NOT NULL,
    random_index bigint NOT NULL,
    ref_epoch double precision NOT NULL,
    ra double precision NOT NULL,
    ra_error double precision NOT NULL,
    "dec" double precision NOT NULL,
    dec_error double precision NOT NULL,
    parallax double precision NOT NULL,
    parallax_error double precision NOT NULL,
    parallax_over_error double precision NOT NULL,
    pm double precision NOT NULL,
    pmra double precision NOT NULL,
    pmra_error double precision NOT NULL,
    pmdec double precision NOT NULL,
    pmdec_error double precision NOT NULL,
    ra_dec_corr double precision NOT NULL,
    ra_parallax_corr double precision NOT NULL,
    ra_pmra_corr double precision NOT NULL,
    ra_pmdec_corr double precision NOT NULL,
    dec_parallax_corr double precision NOT NULL,
    dec_pmra_corr double precision NOT NULL,
    dec_pmdec_corr double precision NOT NULL,
    parallax_pmra_corr double precision NOT NULL,
    parallax_pmdec_corr double precision NOT NULL,
    pmra_pmdec_corr double precision NOT NULL,
    astrometric_n_obs_al bigint NOT NULL,
    astrometric_n_obs_ac bigint NOT NULL,
    astrometric_n_good_obs_al bigint NOT NULL,
    astrometric_n_bad_obs_al bigint NOT NULL,
    astrometric_gof_al double precision NOT NULL,
    astrometric_chi2_al double precision NOT NULL,
    astrometric_excess_noise double precision NOT NULL,
    astrometric_excess_noise_sig double precision NOT NULL,
    astrometric_params_solved bigint NOT NULL,
    astrometric_primary_flag boolean NOT NULL,
    nu_eff_used_in_astrometry double precision NOT NULL,
    pseudocolour double precision NOT NULL,
    pseudocolour_error double precision NOT NULL,
    ra_pseudocolour_corr double precision NOT NULL,
    dec_pseudocolour_corr double precision NOT NULL,
    parallax_pseudocolour_corr double precision NOT NULL,
    pmra_pseudocolour_corr double precision NOT NULL,
    pmdec_pseudocolour_corr double precision NOT NULL,
    astrometric_matched_transits bigint NOT NULL,
    visibility_periods_used bigint NOT NULL,
    astrometric_sigma5d_max double precision NOT NULL,
    matched_transits bigint NOT NULL,
    new_matched_transits bigint NOT NULL,
    matched_transits_removed bigint NOT NULL,
    ipd_gof_harmonic_amplitude double precision NOT NULL,
    ipd_gof_harmonic_phase double precision NOT NULL,
    ipd_frac_multi_peak bigint NOT NULL,
    ipd_frac_odd_win bigint NOT NULL,
    ruwe double precision NOT NULL,
    scan_direction_strength_k1 double precision NOT NULL,
    scan_direction_strength_k2 double precision NOT NULL,
    scan_direction_strength_k3 double precision NOT NULL,
    scan_direction_strength_k4 double precision NOT NULL,
    scan_direction_mean_k1 double precision NOT NULL,
    scan_direction_mean_k2 double precision NOT NULL,
    scan_direction_mean_k3 double precision NOT NULL,
    scan_direction_mean_k4 double precision NOT NULL,
    duplicated_source boolean NOT NULL,
    phot_g_n_obs bigint NOT NULL,
    phot_g_mean_flux double precision NOT NULL,
    phot_g_mean_flux_error double precision NOT NULL,
    phot_g_mean_flux_over_error double precision NOT NULL,
    phot_g_mean_mag double precision NOT NULL,
    phot_bp_n_obs bigint NOT NULL,
    phot_bp_mean_flux double precision NOT NULL,
    phot_bp_mean_flux_error double precision NOT NULL,
    phot_bp_mean_flux_over_error double precision NOT NULL,
    phot_bp_mean_mag double precision NOT NULL,
    phot_rp_n_obs bigint NOT NULL,
    phot_rp_mean_flux double precision NOT NULL,
    phot_rp_mean_flux_error double precision NOT NULL,
    phot_rp_mean_flux_over_error double precision NOT NULL,
    phot_rp_mean_mag double precision NOT NULL,
    phot_bp_rp_excess_factor double precision NOT NULL,
    phot_bp_n_contaminated_transits double precision NOT NULL,
    phot_bp_n_blended_transits double precision NOT NULL,
    phot_rp_n_contaminated_transits double precision NOT NULL,
    phot_rp_n_blended_transits double precision NOT NULL,
    phot_proc_mode double precision NOT NULL,
    bp_rp double precision NOT NULL,
    bp_g double precision NOT NULL,
    g_rp double precision NOT NULL,
    radial_velocity double precision NOT NULL,
    radial_velocity_error double precision NOT NULL,
    rv_method_used double precision NOT NULL,
    rv_nb_transits double precision NOT NULL,
    rv_nb_deblended_transits double precision NOT NULL,
    rv_visibility_periods_used double precision NOT NULL,
    rv_expected_sig_to_noise double precision NOT NULL,
    rv_renormalised_gof double precision NOT NULL,
    rv_chisq_pvalue double precision NOT NULL,
    rv_time_duration double precision NOT NULL,
    rv_amplitude_robust double precision NOT NULL,
    rv_template_teff double precision NOT NULL,
    rv_template_logg double precision NOT NULL,
    rv_template_fe_h double precision NOT NULL,
    rv_atm_param_origin double precision NOT NULL,
    vbroad double precision NOT NULL,
    vbroad_error double precision NOT NULL,
    vbroad_nb_transits double precision NOT NULL,
    grvs_mag double precision NOT NULL,
    grvs_mag_error double precision NOT NULL,
    grvs_mag_nb_transits double precision NOT NULL,
    rvs_spec_sig_to_noise double precision NOT NULL,
    phot_variable_flag character varying(16) NOT NULL,
    l double precision NOT NULL,
    b double precision NOT NULL,
    ecl_lon double precision NOT NULL,
    ecl_lat double precision NOT NULL,
    in_qso_candidates boolean NOT NULL,
    in_galaxy_candidates boolean NOT NULL,
    non_single_star bigint NOT NULL,
    has_xp_continuous boolean NOT NULL,
    has_xp_sampled boolean NOT NULL,
    has_rvs boolean NOT NULL,
    has_epoch_photometry boolean NOT NULL,
    has_epoch_rv boolean NOT NULL,
    has_mcmc_gspphot boolean NOT NULL,
    has_mcmc_msc boolean NOT NULL,
    in_andromeda_survey boolean NOT NULL,
    classprob_dsc_combmod_quasar double precision NOT NULL,
    classprob_dsc_combmod_galaxy double precision NOT NULL,
    classprob_dsc_combmod_star double precision NOT NULL,
    teff_gspphot double precision NOT NULL,
    teff_gspphot_lower double precision NOT NULL,
    teff_gspphot_upper double precision NOT NULL,
    logg_gspphot double precision NOT NULL,
    logg_gspphot_lower double precision NOT NULL,
    logg_gspphot_upper double precision NOT NULL,
    mh_gspphot double precision NOT NULL,
    mh_gspphot_lower double precision NOT NULL,
    mh_gspphot_upper double precision NOT NULL,
    distance_gspphot double precision NOT NULL,
    distance_gspphot_lower double precision NOT NULL,
    distance_gspphot_upper double precision NOT NULL,
    azero_gspphot double precision NOT NULL,
    azero_gspphot_lower double precision NOT NULL,
    azero_gspphot_upper double precision NOT NULL,
    ag_gspphot double precision NOT NULL,
    ag_gspphot_lower double precision NOT NULL,
    ag_gspphot_upper double precision NOT NULL,
    ebpminrp_gspphot double precision NOT NULL,
    ebpminrp_gspphot_lower double precision NOT NULL,
    ebpminrp_gspphot_upper double precision NOT NULL,
    libname_gspphot character varying(8)
);


ALTER TABLE public.gaia_source OWNER TO tglctester;

--
-- Name: gaia_source gaia_source_pkey; Type: CONSTRAINT; Schema: public; Owner: tglctester
--

ALTER TABLE ONLY public.gaia_source
    ADD CONSTRAINT gaia_source_pkey PRIMARY KEY (source_id);


--
-- Name: pointing; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX pointing ON public.gaia_source USING btree (public.q3c_ang2ipix(ra, "dec"));

ALTER TABLE public.gaia_source CLUSTER ON pointing;


--
-- PostgreSQL database dump complete
--

-- Insert data from CSV file
COPY public.gaia_source FROM '/docker-entrypoint-initdb.d/gaia_source.csv' DELIMITER ',' CSV HEADER;
