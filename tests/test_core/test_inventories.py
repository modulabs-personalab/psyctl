"""Tests for the inventory system (base, registry, and all inventory types)."""

from __future__ import annotations

import math

import pytest

from psyctl.data.inventories import (
    INDCOL,
    IPIPNEO,
    REI,
    SD4,
    VGQ,
    create_inventory,
    get_available_inventories,
    get_registry_info,
)

# ---------------------------------------------------------------------------
# Expected inventory metadata
# ---------------------------------------------------------------------------

ALL_INVENTORY_NAMES = ["ipip_neo_120", "rei_40", "sd4_28", "vgq_14", "indcolen_1"]

EXPECTED_QUESTION_COUNTS = {
    "ipip_neo_120": 120,
    "rei_40": 40,
    "sd4_28": 28,
    "vgq_14": 14,
    "indcolen_1": 32,
}

EXPECTED_DOMAINS = {
    "ipip_neo_120": {"N", "E", "O", "A", "C"},
    "rei_40": {"RA", "RE", "EA", "EE", "R", "E"},
    "sd4_28": {"M", "N", "P", "S"},
    "vgq_14": {"VGQ"},
    "indcolen_1": {"HI", "VI", "HC", "VC"},
}


# ===================================================================
# 1. Registry tests
# ===================================================================


class TestRegistry:
    """Tests for the inventory registry functions."""

    @pytest.mark.parametrize("name", ALL_INVENTORY_NAMES)
    def test_create_inventory_all_registered(self, name: str) -> None:
        inv = create_inventory(name)
        assert inv is not None
        assert len(inv.questions) > 0

    def test_create_inventory_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown inventory"):
            create_inventory("nonexistent_99")

    def test_get_available_inventories(self) -> None:
        available = get_available_inventories()
        for name in ALL_INVENTORY_NAMES:
            assert name in available, f"{name} not in available inventories"

    def test_get_registry_info(self) -> None:
        info_list = get_registry_info()
        assert len(info_list) >= len(ALL_INVENTORY_NAMES)
        registry_keys = {item["registry_key"] for item in info_list}
        for name in ["ipip_neo", "rei", "sd4", "vgq", "indcolen"]:
            assert name in registry_keys


# ===================================================================
# 2. BaseInventory shared functionality
# ===================================================================


class TestBaseInventoryShared:
    """Tests for shared BaseInventory methods across all inventories."""

    @pytest.mark.parametrize("name", ALL_INVENTORY_NAMES)
    def test_load_config(self, name: str) -> None:
        inv = create_inventory(name)
        assert isinstance(inv.config, dict)
        assert "domains" in inv.config
        assert "data_file" in inv.config

    @pytest.mark.parametrize(
        "name,expected_count",
        list(EXPECTED_QUESTION_COUNTS.items()),
    )
    def test_load_questions_count(self, name: str, expected_count: int) -> None:
        inv = create_inventory(name)
        assert len(inv.questions) == expected_count

    @pytest.mark.parametrize(
        "name,expected_domains",
        list(EXPECTED_DOMAINS.items()),
    )
    def test_get_supported_traits(self, name: str, expected_domains: set[str]) -> None:
        inv = create_inventory(name)
        traits = inv.get_supported_traits()
        codes = {t["code"] for t in traits}
        assert codes == expected_domains

    @pytest.mark.parametrize("name", ALL_INVENTORY_NAMES)
    def test_get_questions_all(self, name: str) -> None:
        inv = create_inventory(name)
        all_qs = inv.get_questions()
        assert len(all_qs) == len(inv.questions)
        # Should be a copy, not the same list
        assert all_qs is not inv.questions

    @pytest.mark.parametrize(
        "name,trait",
        [
            ("ipip_neo_120", "N"),
            ("rei_40", "RA"),
            ("sd4_28", "M"),
            ("vgq_14", "VGQ"),
            ("indcolen_1", "HI"),
        ],
    )
    def test_get_questions_filtered(self, name: str, trait: str) -> None:
        inv = create_inventory(name)
        filtered = inv.get_questions(trait=trait)
        assert len(filtered) > 0
        assert len(filtered) < len(inv.questions) or len(EXPECTED_DOMAINS[name]) == 1
        for q in filtered:
            assert q.get("domain") == trait or q.get("trait") == trait

    @pytest.mark.parametrize("name", ALL_INVENTORY_NAMES)
    def test_get_inventory_info(self, name: str) -> None:
        inv = create_inventory(name)
        info = inv.get_inventory_info()
        assert "name" in info
        assert "version" in info
        assert "total_questions" in info
        assert "traits" in info
        assert "questions_per_trait" in info
        assert info["total_questions"] == len(inv.questions)

    def test_z_to_percentile_zero(self) -> None:
        inv = create_inventory("vgq_14")
        assert inv._z_to_percentile(0.0) == pytest.approx(50.0)

    def test_z_to_percentile_positive(self) -> None:
        inv = create_inventory("vgq_14")
        expected = 50.0 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
        assert inv._z_to_percentile(1.0) == pytest.approx(expected, rel=1e-4)
        assert inv._z_to_percentile(1.0) == pytest.approx(84.134, abs=0.01)

    def test_z_to_percentile_negative(self) -> None:
        inv = create_inventory("vgq_14")
        expected = 50.0 * (1.0 + math.erf(-1.0 / math.sqrt(2.0)))
        assert inv._z_to_percentile(-1.0) == pytest.approx(expected, rel=1e-4)
        assert inv._z_to_percentile(-1.0) == pytest.approx(15.866, abs=0.01)


# ===================================================================
# 3. calculate_scores
# ===================================================================


class TestCalculateScores:
    """Tests for scoring logic."""

    def test_vgq_simple_scoring(self) -> None:
        """VGQ with 14 items all scoring 3.0 -> raw = 42.0."""
        inv = create_inventory("vgq_14")
        responses = {"VGQ": [3.0] * 14}
        results = inv.calculate_scores(responses)

        assert "VGQ" in results
        vgq = results["VGQ"]
        assert vgq["raw_score"] == pytest.approx(42.0)
        assert vgq["num_items"] == 14.0
        assert vgq["mean_score"] == pytest.approx(3.0)
        # raw=42, mean=42, std=10 -> z=0 -> percentile=50
        assert vgq["z_score"] == pytest.approx(0.0)
        assert vgq["percentile"] == pytest.approx(50.0)

    def test_score_result_fields(self) -> None:
        """Verify all expected fields exist in score results."""
        inv = create_inventory("vgq_14")
        responses = {"VGQ": [3.0] * 14}
        results = inv.calculate_scores(responses)

        expected_fields = {
            "raw_score",
            "z_score",
            "percentile",
            "num_items",
            "domain_name",
            "mean_score",
            "population_mean",
            "population_std",
        }
        assert expected_fields.issubset(set(results["VGQ"].keys()))

    def test_rei_parent_scale_aggregation(self) -> None:
        """REI parent scales R and E aggregate from subscales."""
        inv = create_inventory("rei_40")
        responses = {
            "RA": [3.0] * 10,
            "RE": [4.0] * 10,
            "EA": [2.0] * 10,
            "EE": [5.0] * 10,
        }
        results = inv.calculate_scores(responses)

        # Subscales should be present
        assert "RA" in results
        assert "RE" in results
        assert "EA" in results
        assert "EE" in results

        # Parent scales should aggregate
        assert "R" in results
        r_result = results["R"]
        # R raw_score = sum(RA) + sum(RE) = 30 + 40 = 70
        assert r_result["raw_score"] == pytest.approx(70.0)
        assert r_result["num_items"] == 20.0

        assert "E" in results
        e_result = results["E"]
        # E raw_score = sum(EA) + sum(EE) = 20 + 50 = 70
        assert e_result["raw_score"] == pytest.approx(70.0)
        assert e_result["num_items"] == 20.0

    def test_unknown_domain_ignored(self) -> None:
        """Responses for unknown domains are silently ignored."""
        inv = create_inventory("vgq_14")
        responses = {"VGQ": [3.0] * 14, "UNKNOWN": [1.0, 2.0]}
        results = inv.calculate_scores(responses)
        assert "UNKNOWN" not in results
        assert "VGQ" in results


# ===================================================================
# 4. _normalize_trait
# ===================================================================


class TestNormalizeTrait:
    """Tests for trait normalization in each inventory."""

    @pytest.mark.parametrize(
        "name,full_name,expected_code",
        [
            ("ipip_neo_120", "Neuroticism", "N"),
            ("ipip_neo_120", "Extraversion", "E"),
            ("ipip_neo_120", "Openness", "O"),
            ("ipip_neo_120", "Agreeableness", "A"),
            ("ipip_neo_120", "Conscientiousness", "C"),
            ("rei_40", "Rational Ability", "RA"),
            ("rei_40", "Rational Engagement", "RE"),
            ("rei_40", "Experiential Ability", "EA"),
            ("rei_40", "Experiential Engagement", "EE"),
            ("rei_40", "Rationality", "R"),
            ("rei_40", "Experientiality", "E"),
            ("sd4_28", "Machiavellianism", "M"),
            ("sd4_28", "Narcissism", "N"),
            ("sd4_28", "Psychopathy", "P"),
            ("sd4_28", "Sadism", "S"),
            ("vgq_14", "Gaslighting Victimization", "VGQ"),
            ("indcolen_1", "Horizontal Individualism", "HI"),
            ("indcolen_1", "Vertical Individualism", "VI"),
            ("indcolen_1", "Horizontal Collectivism", "HC"),
            ("indcolen_1", "Vertical Collectivism", "VC"),
        ],
    )
    def test_normalize_full_name(
        self, name: str, full_name: str, expected_code: str
    ) -> None:
        inv = create_inventory(name)
        assert inv._normalize_trait(full_name) == expected_code

    @pytest.mark.parametrize(
        "name,code",
        [
            ("ipip_neo_120", "N"),
            ("rei_40", "RA"),
            ("sd4_28", "M"),
            ("vgq_14", "VGQ"),
            ("indcolen_1", "HI"),
        ],
    )
    def test_normalize_code_identity(self, name: str, code: str) -> None:
        inv = create_inventory(name)
        assert inv._normalize_trait(code) == code

    @pytest.mark.parametrize(
        "name,bad_trait",
        [
            ("ipip_neo_120", "InvalidTrait"),
            ("rei_40", "Magic"),
            ("sd4_28", "Kindness"),
            ("vgq_14", "NotATrait"),
            ("indcolen_1", "BadTrait"),
        ],
    )
    def test_normalize_unknown_raises(self, name: str, bad_trait: str) -> None:
        inv = create_inventory(name)
        with pytest.raises(ValueError, match="Unrecognized trait"):
            inv._normalize_trait(bad_trait)


# ===================================================================
# 5. SD4 config_key override
# ===================================================================


class TestSD4ConfigKey:
    """Tests for SD4 config_key override."""

    def test_sd4_uses_sd4_config_key(self) -> None:
        """SD4 should use 'sd4' as config key, not 'sd4_28'."""
        assert SD4.config_key == "sd4"

    def test_sd4_resolves_config_key(self) -> None:
        inv = SD4()
        assert inv._resolve_config_key() == "sd4"

    def test_sd4_loads_successfully(self) -> None:
        inv = create_inventory("sd4_28")
        assert inv.config["name"] == "SD4"
        assert len(inv.questions) == 28


# ===================================================================
# 6. Edge cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_create_sd4_without_version(self) -> None:
        """sd4 (without version suffix) should use default version."""
        inv = create_inventory("sd4")
        assert inv.version == "28"
        assert len(inv.questions) == 28

    def test_inventory_classes_directly(self) -> None:
        """Inventory classes can be instantiated directly."""
        assert IPIPNEO().version == "120"
        assert REI().version == "40"
        assert SD4().version == "28"
        assert VGQ().version == "14"
        assert INDCOL().version == "1"

    def test_rei_get_questions_parent_scale_r(self) -> None:
        """REI get_questions with parent trait 'R' returns RA+RE questions."""
        inv = create_inventory("rei_40")
        r_qs = inv.get_questions(trait="R")
        domains_in_result = {q["domain"] for q in r_qs}
        assert domains_in_result == {"RA", "RE"}

    def test_rei_get_questions_parent_scale_e(self) -> None:
        """REI get_questions with parent trait 'E' returns EA+EE questions."""
        inv = create_inventory("rei_40")
        e_qs = inv.get_questions(trait="E")
        domains_in_result = {q["domain"] for q in e_qs}
        assert domains_in_result == {"EA", "EE"}
