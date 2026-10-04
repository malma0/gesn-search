from pathlib import Path

from gesn.parse import join_name, parse

XML = """<?xml version="1.0" encoding="utf-8"?>
<base>
  <ResourcesDirectory>
    <ResourceCategory Type="СТРОИТЕЛЬНЫЕ РАБОТЫ" CodePrefix="ГЭСН">
      <Section Name="Отделочные работы" Type="Сборник" Code="15">
        <Section Name="ОКРАСКА" Type="Раздел" Code="4">
          <Section Name="Окраска водоэмульсионными составами" Type="Таблица" Code="15-04-005">
            <NameGroup BeginName="Окраска   водоэмульсионными составами:">
              <Work Code="15-04-005-01" EndName="стен" MeasureUnit="100 м2">
                <Content>
                  <Item Text="Шлифовка подмазанных мест." />
                  <Item Text="Окраска поверхностей." />
                </Content>
                <Resources><Resource Code="2" Quantity="0.09" /></Resources>
              </Work>
            </NameGroup>
          </Section>
        </Section>
      </Section>
      <Section Name="Полы" Type="Сборник" Code="11">
        <Section Name="Стяжки" Type="Таблица" Code="11-01-011">
          <NameGroup BeginName="Устройство стяжек цементных толщиной 20 мм">
            <Work Code="11-01-011-01" EndName="" MeasureUnit="100 м2" />
          </NameGroup>
        </Section>
      </Section>
    </ResourceCategory>
  </ResourcesDirectory>
</base>
"""


def write_xml(tmp_path: Path) -> Path:
    path = tmp_path / "gesn.xml"
    path.write_text(XML, encoding="utf-8")
    return path


def test_parse_flattens_hierarchy(tmp_path):
    df = parse(write_xml(tmp_path))
    row = df.set_index("code").loc["15-04-005-01"]

    assert row["collection_code"] == "15"
    assert row["section_name"] == "ОКРАСКА"
    assert row["subsection_name"] == ""
    assert row["table_code"] == "15-04-005"
    assert row["full_name"] == "Окраска водоэмульсионными составами: стен"
    assert row["unit"] == "100 м2"
    assert row["content"] == "Шлифовка подмазанных мест. Окраска поверхностей."


def test_parse_filters_collections(tmp_path):
    df = parse(write_xml(tmp_path), ["11"])
    assert df["code"].tolist() == ["11-01-011-01"]


def test_join_name():
    assert join_name("Стяжка толщиной:", "") == "Стяжка толщиной"
    assert join_name("", "стен") == "стен"
