# TGLC Testing Sample Databases

This directory contains database data dumped from the infrastructure at the MIT TESS Science Office, which is managed using [pyticdb](https://github.com/mit-kavli-institute/pyticdb). There are three tables used by TGLC: the main TIC and Gaia DR3 tables, as well as the [Gaia DR3 "DR2 Neighborhood" table](https://gaia.aip.de/metadata/gaiaedr3/dr2_neighbourhood/), which has matches between Gaia DR2 and DR3 source IDs. This directory contains SQL schema definitions and CSV data files for these tables as they exist on the TSO servers.

## Schema definitions

The schema definitions were created using `pg_dump`.

```shell
pg_dump -U $USER -h $DB_HOST -p $TIC_DB_PORT -d tic_82 -t ticentries --schema-only > ticentries.sql
pg_dump -U $USER -h $DB_HOST -p $GAIA_DB_PORT -d gaia_dr3 -t gaia_source --schema-only > gaia_source.sql
pg_dump -U $USER -h $DB_HOST -p $TIC_DB_PORT -d tic_82 -t dr2_to_dr3 --schema-only > dr2_to_dr3.sql
```

The schema definitions were modified to grant permissions to the `tglctester` user rather than the actual users from the TSO servers, since the `tglctester` is used in the TGLC test suite. Any `GRANT` statements for users on TSO servers were removed.

## Data

The data files were created using `psql` and `\copy`. Data was dumped for the end-to-end tests, which use orbit 185, camera 1, CCD 1, cutout 0,0. The actual TGLC pipeline had been run for this orbit, so a list of TIC IDs and Gaia source IDs (DR2 and DR3) were created for this cutout by loading the cutout's `Source` object and using its `.tic` and `.gaia` catalogs. These lists were stored as files `tic_ids.txt`, `gaia_dr2_source_ids.txt`, and `gaia_dr3_source_ids.txt` with one ID per line and no header. Then, variations on the following set of commands were run: create a temporary table containing the relevant IDs, select all data in those rows, and copy that selection to a CSV file. For example, on the Gaia (DR3) database:

```sql
CREATE TEMP TABLE source_id_list (source_id bigint);
\copy source_id_list FROM 'gaia_dr3_source_ids.txt' WITH (FORMAT CSV);
\copy (SELECT * FROM gaia_source where id IN (SELECT source_id FROM source_id_list)) TO 'gaia_source.csv' WITH (FORMAT CSV, HEDER);
```

Finally, `COPY` statements were added to the `.sql` files to copy the data from the CSVs into the databases when they are set up in pytest fixtures.

```sql
-- Insert data from CSV file
COPY public.gaia_source FROM 'gaia_source.csv' DELIMITER ',' CSV HEADER;
```
