from datetime import datetime, timedelta

from django.test import TestCase
from django.utils import timezone

from api.models import StationInfo, WaterInfo, WarningNotice
from .tasks import insert_water_data, update_predict


class WarningNoticeTestCase(TestCase):
    def setUp(self):
        StationInfo.objects.create(
            id='63000200',
            name='桥东村',
            city='杭州',
            county='临安',
            guaranteed=85.66,
            warning=84.66
        )

        start_time = datetime.strptime("2021-06-24 6:00:00", "%Y-%m-%d %H:%M:%S")

        for i in range(12):
            WaterInfo.objects.create(
                station_id='63000200',
                times=timezone.make_aware(start_time + timedelta(hours=i)),
                rains=0,
                waterlevels=88.5
            )

        self.data = {
            'times': "2021-06-24 18:00:00",
            'station_id': '63000200',
            'rains': 0,
            'waterlevels': 88.5

        }

    def test_insert_test(self):
        insert_water_data(self.data)
        update_predict('63000200')
        self.assertEqual(WarningNotice.objects.count(), 1)
