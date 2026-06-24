"""Tests for xclass.catalog — training dataset assembly and deduplication.

All tests are network-free; external calls are mocked via conftest fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# normalize_name tests
# ---------------------------------------------------------------------------



def test_normalize_name_nan():
    """normalize_name(NaN) should return an empty string."""
    from xclass.catalog import normalize_name

    assert normalize_name(float("nan")) == ""
    assert normalize_name(None) == ""



def test_normalize_name_empty():
    """normalize_name('') and normalize_name('   ') should return ''."""
    from xclass.catalog import normalize_name

    assert normalize_name("") == ""
    assert normalize_name("   ") == ""



def test_normalize_name_special_chars():
    """Non-alphanumeric characters are stripped; result is uppercase."""
    from xclass.catalog import normalize_name

    assert normalize_name("Sco X-1") == "SCOX1"
    assert normalize_name("2CXO J161546.5-232327") == "2CXOJ1615465232327"



def test_normalize_name_unicode():
    """Unicode input is handled gracefully (uppercased where possible)."""
    from xclass.catalog import normalize_name

    result = normalize_name("Müller-10")
    # Should not raise; exact output depends on implementation
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# classify_sptype tests
# ---------------------------------------------------------------------------



def test_classify_sptype_ob_types():
    """O and B spectral types -> 'HM-STAR'."""
    from xclass.catalog import classify_sptype

    assert classify_sptype("O5V") == "HM-STAR"
    assert classify_sptype("B2Ib") == "HM-STAR"
    assert classify_sptype("O9III") == "HM-STAR"
    assert classify_sptype("B0") == "HM-STAR"



def test_classify_sptype_afgkm_types():
    """A, F, G, K, M spectral types -> 'LM-STAR'."""
    from xclass.catalog import classify_sptype

    assert classify_sptype("A0V") == "LM-STAR"
    assert classify_sptype("F5") == "LM-STAR"
    assert classify_sptype("G2V") == "LM-STAR"
    assert classify_sptype("K5V") == "LM-STAR"
    assert classify_sptype("M0") == "LM-STAR"



def test_classify_sptype_nan():
    """NaN and empty string -> None."""
    from xclass.catalog import classify_sptype

    assert classify_sptype(float("nan")) is None
    assert classify_sptype("") is None
    assert classify_sptype(None) is None


# ---------------------------------------------------------------------------
# UnionFind tests
# ---------------------------------------------------------------------------



def test_union_find_merge():
    """After union(0, 1) the two elements share a root."""
    from xclass.catalog import UnionFind

    uf = UnionFind(5)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    # Elements 2, 3, 4 are still separate
    assert uf.find(2) != uf.find(0)



def test_union_find_path_compression():
    """find() should be idempotent; repeated calls return the same root."""
    from xclass.catalog import UnionFind

    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    root = uf.find(3)
    # After path compression, find(0) should also equal root
    assert uf.find(0) == root



def test_union_find_groups():
    """groups() returns a dict mapping each root to its member list."""
    from xclass.catalog import UnionFind

    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    groups = uf.groups()
    # Two groups of size 2
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [2, 2]


# ---------------------------------------------------------------------------
# build_master_resolved_table tests
# ---------------------------------------------------------------------------



def test_build_master_resolved_dedup_by_name(synthetic_td_df):
    """Sources with identical normalised names in the same class are merged."""
    from xclass.catalog import build_master_resolved_table

    # Add a duplicate row (same name, different capitalisation)
    extra = synthetic_td_df.iloc[[0]].copy()
    extra["name"] = "SCO X-1"  # same as 'Sco X-1' after normalisation
    df = pd.concat([synthetic_td_df, extra], ignore_index=True)

    result = build_master_resolved_table(df, match_radius_arcsec=5.0)
    lmxb_rows = result[result["Class"] == "LMXB"]
    # The two Sco X-1 entries should merge into one master row
    assert len(lmxb_rows) == 1
    assert lmxb_rows.iloc[0]["n_input_rows"] == 2



def test_build_master_resolved_dedup_by_position(synthetic_td_df):
    """Sources within 5 arcsec in the same class are merged."""
    from xclass.catalog import build_master_resolved_table

    # Create two sources of the same class within 1 arcsec of each other
    df = pd.DataFrame(
        {
            "name": ["Source_A", "Source_B"],
            "Class": ["LMXB", "LMXB"],
            "ra": [100.0000, 100.0001],   # ~0.36 arcsec apart in RA at dec=0
            "dec": [0.0, 0.0],
            "source_catalog": ["CatA", "CatB"],
            "source_ref": ["RefA", "RefB"],
            "label_confidence": [1.0, 1.0],
        }
    )
    result = build_master_resolved_table(df, match_radius_arcsec=5.0)
    assert len(result) == 1
    assert result.iloc[0]["n_input_rows"] == 2
