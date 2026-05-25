# tests/test_enums.py
"""app/models/enums.py 의 Category / UserStatus 와 Category._cat_name 검증."""

from app.models.enums import Category, UserStatus


class TestCategory:
    def test_is_str_enum(self):
        # StrEnum 이므로 값 자체가 문자열과 동일하게 취급됨
        assert Category.MUSIC == "음악"
        assert Category.MUSIC.value == "음악"
        assert str(Category.MUSIC) == "음악"

    def test_lookup_by_value(self):
        assert Category("게임") is Category.GAME
        assert Category("스터디") is Category.STUDY

    def test_all_members_unique(self):
        values = [c.value for c in Category]
        assert len(values) == len(set(values))  # @unique 보장


class TestCatName:
    def test_none_returns_empty_string(self):
        assert Category._cat_name(None) == ""

    def test_enum_member_returns_value(self):
        assert Category._cat_name(Category.MUSIC) == "음악"
        assert Category._cat_name(Category.GAME) == "게임"

    def test_plain_string_passthrough(self):
        assert Category._cat_name("음악") == "음악"

    def test_other_type_is_stringified(self):
        assert Category._cat_name(123) == "123"


class TestUserStatus:
    def test_members(self):
        assert UserStatus.JOIN == "JOIN"
        assert UserStatus.CLICK == "CLICK"

    def test_lookup_by_value(self):
        assert UserStatus("JOIN") is UserStatus.JOIN
        assert UserStatus("CLICK") is UserStatus.CLICK
