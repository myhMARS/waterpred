from django.db import migrations
from api.models import StationInfo


def init_stations(apps, schema_editor):
    StationInfo.objects.create(
        id="63000120",
        name="里畈东坑溪",
        city="杭州",
        county="临安县",
    )
    StationInfo.objects.create(
        id="63000100",
        name="里畈水库",
        city="杭州",
        county="临安县",
        flood_limit=234.73
    )
    StationInfo.objects.create(
        id="63000200",
        name="桥东村",
        guaranteed=85.66,
        warning=84.66
    )


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(init_stations),
    ]
