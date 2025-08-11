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
-- Name: dr2_to_dr3; Type: TABLE; Schema: public; Owner: tglctester
--

CREATE TABLE public.dr2_to_dr3 (
    dr2_source_id bigint,
    dr3_source_id bigint
);


ALTER TABLE public.dr2_to_dr3 OWNER TO tglctester;

--
-- Name: dr2_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX dr2_idx ON public.dr2_to_dr3 USING btree (dr2_source_id);


--
-- Name: dr3_idx; Type: INDEX; Schema: public; Owner: tglctester
--

CREATE INDEX dr3_idx ON public.dr2_to_dr3 USING btree (dr3_source_id);


--
-- PostgreSQL database dump complete
--

-- Insert data from CSV file
COPY public.dr2_to_dr3 FROM '/docker-entrypoint-initdb.d/dr2_to_dr3.csv' DELIMITER ',' CSV HEADER;

