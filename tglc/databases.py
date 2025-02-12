"""
This module provides access to databases through the Pyticdb package, which is used at the TSO to
keep local copies of databases like the TIC (TESS Input Catalog) and Gaia.
"""

import csv
from os import PathLike
import typing

import numpy as np
import pandas as pd
from pyticdb import Databases
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert


class Database:
    """Mixin class for reflecting a database from Pyticdb."""

    def __init__(self, database_name: str):
        self.meta, self.Session = Databases[database_name]

    def table(self, tablename: str) -> sa.Table:
        return self.meta.tables[tablename]

    def column(self, tablename: str, field: str):
        return self.table(tablename).c[field]

    def select(self, tablename: str, *fields: str, **annotations):
        columns = [self.column(tablename, field) for field in fields]

        for label, expression in annotations.items():
            columns.append(expression.label(label))

        return sa.select(*columns)

    def join_args(self, left_table: str, right_table: str, left_col: str, right_col: str):
        l_table = self.table(left_table)
        r_table = self.table(right_table)

        left_col = getattr(l_table.c, left_col)
        right_col = getattr(r_table.c, right_col)

        return r_table, left_col == right_col

    def by_id(self, tablename: str, id: typing.Any):
        table = self.table(tablename)

        # For now, no composite primary keys are allowed
        columns = tuple(table.primary_key.columns)

        if len(columns) == 1:
            if isinstance(id, str):
                return columns[0] == id
            try:
                ids = list(id)
                return columns[0].in_(ids)
            except TypeError:
                return columns[0] == id
        else:
            # Not supporting composite primary keys yet
            raise NotImplementedError

    def execute(self, query):
        with self.Session() as db:
            return list(db.execute(query).fetchall())

    def to_df(self, query):
        results = self.execute(query)
        top_row = results[0]
        return pd.DataFrame(results, columns=top_row._fields)

    def query_by_id(
        self, tablename: str, id: typing.Any, *fields: str, as_query=False, **annotations
    ):
        q = self.select(tablename, *fields, **annotations)
        q = q.where(self.by_id(tablename, id))
        if as_query:
            return q
        return self.execute(q)


class Q3C_Mixin:
    """
    A Database mixin to describe additional database capabilities for the q3c
    spatial extension for PostgreSQL.
    """

    table: typing.Callable  # Forward declare
    select: typing.Callable
    execute: typing.Callable

    def distance_to(self, tablename: str, ref_ra: float, ref_dec: float):
        table = self.table(tablename)
        col = sa.func.q3c_dist(table.c.ra, table.c.dec, ref_ra, ref_dec)
        return col

    def in_cone(self, tablename: str, ra: float, dec: float, width: float):
        table = self.table(tablename)
        ra_field = table.c.ra
        dec_field = table.c.dec
        return sa.func.q3c_radial_query(ra_field, dec_field, ra, dec, width)

    def query_by_loc(
        self,
        tablename: str,
        ra: float,
        dec: float,
        width: float,
        *fields: str,
        as_query=False,
        **annotations,
    ):
        """Run a cone search based on an RA/Dec sky position."""
        q = self.select(tablename, *fields, **annotations)
        q = q.where(self.in_cone(tablename, ra, dec, width))
        if as_query:
            return q
        return self.execute(q)


class TIC(Database, Q3C_Mixin):
    pass


class Gaia(Database, Q3C_Mixin):
    """
    This class wraps common Gaia queries and capabilities.
    """

    def crossmatch(
        self,
        tic_ra: float,
        tic_dec: float,
        search_width: float,
        tmag: float,
        mag_tolerance: float = 0.1,
    ):
        """
        Find nearby stars that are close, in terms of brightness, to
        a given tmag. The results are ordered from closest to farthest
        in terms of angular distance.

        Parameters
        ----------
        tic_ra: float
            The right ascension coordinate
        tic_dec: float
            The declination coordinate
        search_width: float
            The angular distance which will be used for the cone search.
            This field dramatically influences query performance; if
            query performance is poor, try reducing the search width lower.
        tmag: float
            The reference TESS magnitude to use to match a star.
        mag_tolerance: Optional[float]
            How much tolerance to match TESS magntiudes with GAIA
            magnitudes.

        Returns
        -------
        list[tuple(int, float)]
            Returns a list of tuples ordered by distance. The tuples
            contain two fields: (source_id, distance).
        """
        if np.isnan(tic_ra) or np.isnan(tic_dec):
            raise ValueError("RA and DEC must be defined floats")

        gaia_rmag = self.column("gaia_source", "phot_rp_mean_mag")
        gaia_bmag = self.column("gaia_source", "phot_bp_mean_mag")

        gaia_mag_diff = gaia_bmag - gaia_rmag

        red_magnitude_error = sa.case((gaia_mag_diff > 0.7, 0.1), else_=0.0)

        mag_tolerance_cutoff = sa.func.ABS(gaia_rmag - tmag) < red_magnitude_error + mag_tolerance

        q = self.select(
            "gaia_source", "source_id", distance=self.distance_to("gaia_source", tic_ra, tic_dec)
        )
        q = q.join(
            *self.join_args("gaia_source", "astrophysical_parameters", "source_id", "source_id")
        )

        q = q.where(mag_tolerance_cutoff)
        q = q.where(self.in_cone("gaia_source", tic_ra, tic_dec, search_width))
        q = q.order_by(self.distance_to("gaia_source", tic_ra, tic_dec).desc())
        return self.execute(q)

    def crossmatch_to_tic_id(
        self,
        tic_id: typing.Union[str, int],
        search_width: float,
        tic_db: typing.Optional[str] = None,
        mag_tolerance: float = 0.1,
    ):
        """
        A helper method which will call ``crossmatch`` but instead
        referenceing a TIC id. Subsequent searches on the TIC will
        be done in order to gather the needed information for a
        crossmatch.

        Parameters
        ----------
        tic_id: int
            The TIC identifier.
        search_width: float
            The angular distance which will be used for the cone search.
            This field dramatically influences query performance; if
            query performance is poor, try reducing the search width lower.
        tic_db: Optional[str]
            By default the crossmatch will be done in reference to
            ``tic_82`` but this can be overridden to any TIC catalog
            so long as it has been provisioned.
        mag_tolerance: Optional[float]
            How much tolerance to match TESS magntiudes with GAIA
            magnitudes.

        Returns
        -------
        list[tuple(int, float)]
            Returns a list of tuples ordered by distance. The tuples
            contain two fields: (source_id, distance).
        """

        tic = TIC("tic_82" if tic_db is None else tic_db)
        ra, dec, tmag = tic.query_by_id("ticentries", tic_id, "ra", "dec", "tmag")[0]
        return self.crossmatch(ra, dec, search_width, tmag, mag_tolerance=mag_tolerance)


class Stelvar(Database, Q3C_Mixin):
    def update_table(self, csv_file: PathLike):
        with open(csv_file) as fin:
            reader = csv.DictReader(fin)
            payload = list(reader)

        target_table = self.table("stelvar_1")
        q = insert(target_table).values(payload).on_conflict_do_update()

        with self.Session() as db:
            db.execute(q)
            db.commit()
