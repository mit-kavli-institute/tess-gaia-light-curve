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
-- Name: ticentries; Type: TABLE; Schema: public; Owner: tglctester
--

CREATE TABLE public.ticentries (
    id bigint NOT NULL,
    version bigint,
    hip bigint,
    tyc text,
    ucac text,
    twomass text,
    sdss text,
    allwise text,
    gaia text,
    apass text,
    kic bigint,
    objtype text,
    typesrc text,
    ra double precision,
    "dec" double precision,
    posflag text,
    pmra double precision,
    e_pmra double precision,
    pmdec double precision,
    e_pmdec double precision,
    pmflag text,
    plx double precision,
    e_plx double precision,
    parflag text,
    gallong double precision,
    gallat double precision,
    eclong double precision,
    eclat double precision,
    bmag real,
    e_bmag real,
    vmag real,
    e_vmag real,
    umag real,
    e_umag real,
    gmag real,
    e_gmag real,
    rmag real,
    e_rmag real,
    imag real,
    e_imag real,
    zmag real,
    e_zmag real,
    jmag real,
    e_jmag real,
    hmag real,
    e_hmag real,
    kmag real,
    e_kmag real,
    twomflag text,
    prox real,
    w1mag real,
    e_w1mag real,
    w2mag real,
    e_w2mag real,
    w3mag real,
    e_w3mag real,
    w4mag real,
    e_w4mag real,
    gaiamag real,
    e_gaiamag real,
    tmag real,
    e_tmag real,
    tessflag text,
    spflag text,
    teff real,
    e_teff real,
    logg real,
    e_logg real,
    mh real,
    e_mh real,
    rad real,
    e_rad real,
    mass real,
    e_mass real,
    rho real,
    e_rho real,
    lumclass text,
    lum real,
    e_lum real,
    d real,
    e_d real,
    ebv real,
    e_ebv real,
    numcont bigint,
    contratio real,
    disposition text,
    duplicate_id bigint,
    priority real,
    eneg_ebv real,
    epos_ebv real,
    ebvflag text,
    eneg_mass real,
    epos_mass real,
    eneg_rad real,
    epos_rad real,
    eneg_rho real,
    epos_rho real,
    eneg_logg real,
    epos_logg real,
    eneg_lum real,
    epos_lum real,
    eneg_dist real,
    epos_dist real,
    distflag text,
    eneg_teff real,
    epos_teff real,
    teffflag text,
    gaiabp real,
    e_gaiabp real,
    gaiarp real,
    e_gaiarp real,
    gaiaqflag bigint,
    starchareflag text,
    vmagflag text,
    bmagflag text,
    splists text,
    e_ra double precision,
    e_dec double precision,
    ra_orig double precision,
    dec_orig double precision,
    e_ra_orig double precision,
    e_dec_orig double precision,
    raddflag bigint,
    wdflag bigint,
    gaia3 bigint
);


ALTER TABLE public.ticentries OWNER TO tglctester;

--
-- Name: ticentries ticentries_pkey; Type: CONSTRAINT; Schema: public; Owner: tglctester
--

ALTER TABLE ONLY public.ticentries
    ADD CONSTRAINT ticentries_pkey PRIMARY KEY (id);


--
-- Name: bright_tmag_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX bright_tmag_idx ON public.ticentries USING btree (tmag) WHERE (tmag <= (13.5)::double precision);


--
-- Name: gaia2_int_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX gaia2_int_idx ON public.ticentries USING btree (((gaia)::bigint));


--
-- Name: gaia_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX gaia_idx ON public.ticentries USING btree (gaia);


--
-- Name: id_float_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX id_float_idx ON public.ticentries USING btree (((id)::numeric));


--
-- Name: id_index; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX id_index ON public.ticentries USING btree (id) WITH (fillfactor='100');


--
-- Name: pointing_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX pointing_idx ON public.ticentries USING btree (public.q3c_ang2ipix(ra, "dec"));

ALTER TABLE public.ticentries CLUSTER ON pointing_idx;


--
-- Name: ticentry_gaia3_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX ticentry_gaia3_idx ON public.ticentries USING btree (gaia3) WHERE (gaia3 IS NOT NULL);


--
-- Name: tmag_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX tmag_idx ON public.ticentries USING btree (tmag);


--
-- PostgreSQL database dump complete
--

-- Insert data from CSV file
COPY public.ticentries FROM '/docker-entrypoint-initdb.d/ticentries.csv' DELIMITER ',' CSV HEADER;

