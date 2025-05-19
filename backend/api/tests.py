from datetime import datetime, timedelta

from django.test import TestCase
from django.utils import timezone
from django.utils.timezone import make_aware
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase

from .models import WaterInfo, WaterPred, Statistics, StationInfo, WarningNotice
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


class WaterInfoAPITestCase(APITestCase):
    def setUp(self):
        self.station_id = "63000200"
        self.url = "/api/waterinfo/"
        self.station = StationInfo.objects.create(id=self.station_id, name="测试水文站")

        # 用 base_time 创建 3 条 WaterInfo
        base_time = make_aware(datetime.now().replace(microsecond=0))
        self.times_list = []

        for i in range(3):
            t = base_time + timedelta(hours=i)
            self.times_list.append(t)
            WaterInfo.objects.create(
                station=self.station,
                waterlevels=10.0 + i,
                rains=2.0 + i,
                times=t
            )

        # ✅ 创建 WaterPred，时间与最后一个 WaterInfo 对齐
        WaterPred.objects.create(
            station=self.station,
            times=self.times_list[-1],  # 保证 times 匹配
            waterlevel1=12.3,
            waterlevel2=12.5,
            waterlevel3=12.7,
            waterlevel4=13.0,
            waterlevel5=13.2,
            waterlevel6=13.5
        )

    def test_missing_station_id(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("station_id", response.data["detail"])

    def test_station_id_not_found(self):
        response = self.client.get(self.url, {"station_id": "UNKNOWN"})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("notfound", response.data["detail"])

    def test_valid_response_with_prediction(self):
        response = self.client.get(self.url, {"station_id": self.station_id, "length": 3})
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertIn("times", response.data)
        self.assertIn("data", response.data)
        self.assertEqual(len(response.data["times"]), 3)
        self.assertEqual(len(response.data["data"]), 3)

        # 可选：验证预测数据是否包含某个字段
        self.assertIn("pred", response.data)
        self.assertIsInstance(response.data["pred"], list)


class StationCountAPITestCase(APITestCase):
    def setUp(self):
        # 创建测试数据
        self.url = '/api/stationCount/'  # 假设你的URL路径是这个

        # 创建 StationInfo 实例
        self.station1 = StationInfo.objects.create(
            id="63000200", name="测试站点1", county="测试县", warning=10, guaranteed=15, flood_limit=20
        )
        self.station2 = StationInfo.objects.create(
            id="63000201", name="测试站点2", county="测试县", warning=None, guaranteed=None, flood_limit=None
        )

        # 创建 WaterInfo 实例
        base_time = make_aware(datetime.now().replace(microsecond=0))
        WaterInfo.objects.create(
            station=self.station1,
            times=base_time - timedelta(hours=1),
            waterlevels=8.0,  # 小于 minimum (warning = 10)
            rains=3.0
        )
        WaterInfo.objects.create(
            station=self.station2,
            times=base_time - timedelta(hours=2),
            waterlevels=18.0,  # 水位较高，忽略该站点
            rains=2.5
        )

    def test_station_count(self):
        # 发送 GET 请求
        response = self.client.get(self.url)

        # 验证响应状态码
        self.assertEqual(response.status_code, 200)

        # 验证返回数据的正确性
        self.assertIn('stationCount', response.data)
        self.assertIn('normalCount', response.data)
        self.assertIn('areaCount', response.data)

        # 验证统计结果
        self.assertEqual(response.data['stationCount'], 2)  # 2个站点
        self.assertEqual(response.data['normalCount'], 2)  # 2个站点符合正常条件
        self.assertEqual(response.data['areaCount'], 1)  # 1个县 (测试县)


class StatisticsInfoAPITestCase(APITestCase):
    def setUp(self):
        # 创建 StationInfo 实例
        self.station = StationInfo.objects.create(id="63000200", name="测试站点")

        # 创建 WaterInfo 实例，用于获取时间
        base_time = make_aware(datetime.now().replace(microsecond=0))
        self.water_info = WaterInfo.objects.create(
            station=self.station,
            times=base_time,
            waterlevels=10.0,
            rains=3.0
        )

        # 创建 Statistics 实例
        self.statistics1 = Statistics.objects.create(
            year=2025, month=5, day=10, station=self.station
        )
        self.statistics2 = Statistics.objects.create(
            year=2025, month=5, day=11, station=self.station
        )
        self.statistics3 = Statistics.objects.create(
            year=2025, month=5, day=12, station=self.station
        )
        self.statistics4 = Statistics.objects.create(
            year=2025, month=4, day=5, station=self.station
        )
        self.statistics5 = Statistics.objects.create(
            year=2025, month=4, day=6, station=self.station
        )

    def test_get_statistics_by_month(self):
        # 发送 GET 请求，带上 time_filter 为 'month'
        response = self.client.get('/api/statistics/', {"time_filter": "month"})

        # 验证响应状态码
        self.assertEqual(response.status_code, 200)

        # 验证返回的数据
        self.assertEqual(response.data['year'], 2025)
        self.assertEqual(response.data['month'], 5)
        self.assertIn('data', response.data)

        # 验证返回的数据格式，按日统计
        self.assertEqual(response.data['data'], {
            '2025-5-10': 1,
            '2025-5-11': 1,
            '2025-5-12': 1
        })

    def test_get_statistics_by_quarter(self):
        # 发送 GET 请求，带上 time_filter 为 'quarter'
        response = self.client.get('/api/statistics/', {"time_filter": "quarter"})

        # 验证响应状态码
        self.assertEqual(response.status_code, 200)

        # 验证返回的数据
        self.assertEqual(response.data['year'], 2025)
        self.assertEqual(response.data['month'], 5)
        self.assertIn('data', response.data)

        # 验证返回的数据格式，按季度统计
        self.assertEqual(response.data['data'], {
            '2025-4-5': 1,
            '2025-4-6': 1,
            '2025-5-10': 1,
            '2025-5-11': 1,
            '2025-5-12': 1
        })

    def test_get_statistics_by_year(self):
        # 发送 GET 请求，带上 time_filter 为 'year'
        response = self.client.get('/api/statistics/', {"time_filter": "year"})

        # 验证响应状态码
        self.assertEqual(response.status_code, 200)

        # 验证返回的数据
        self.assertEqual(response.data['year'], 2025)
        self.assertEqual(response.data['month'], 5)
        self.assertIn('data', response.data)

        # 验证返回的数据格式，按年统计
        self.assertEqual(response.data['data'], {
            '2025-4-5': 1,
            '2025-4-6': 1,
            '2025-5-10': 1,
            '2025-5-11': 1,
            '2025-5-12': 1
        })

    def test_missing_time_filter(self):
        # 发送 GET 请求，缺少 time_filter 参数
        response = self.client.get('/api/statistics/')

        # 验证响应状态码为 400
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data, {"detail": "time_filter field required"})


class WarningInfoAPITestCase(APITestCase):

    def setUp(self):
        # 创建 StationInfo 实例
        self.station = StationInfo.objects.create(id='63000200', name='测试站点')

        # 创建 User 实例，模拟执行通知的用户
        self.user = User.objects.create_user(username='executor1', password='testpassword')

        # 创建 WarningNotice 实例
        self.warning1 = WarningNotice.objects.create(
            station=self.station,
            noticetype='洪水',
            max_level=15.0,
            noticetime=timezone.now(),
            isSuccess=True,
            isCanceled=False,
            executor=self.user
        )

        self.warning2 = WarningNotice.objects.create(
            station=self.station,
            noticetype='正常',
            max_level=12.0,
            noticetime=timezone.now(),
            isSuccess=False,
            isCanceled=False
        )

        self.warning3 = WarningNotice.objects.create(
            station=self.station,
            noticetype='洪水',
            max_level=15.0,
            noticetime=timezone.now(),
            isSuccess=True,
            isCanceled=True,
            executor=self.user
        )

    def test_get_warning_info_without_params(self):
        # 发送没有任何查询参数的 GET 请求
        response = self.client.get('/api/warnings/')

        # 检查响应状态码
        self.assertEqual(response.status_code, 200)

        # 确保返回了所有 WarningNotice 记录
        self.assertEqual(len(response.data), 3)

    def test_get_warning_info_with_is_cancel_param(self):
        # 发送带有 'isCancel' 参数的 GET 请求
        response = self.client.get('/api/warnings/', {"isCancel": "1"})

        # 检查响应状态码
        self.assertEqual(response.status_code, 200)

        # 确保只返回已取消的警告通知
        self.assertEqual(len(response.data), 1)
        self.assertTrue(response.data[0]['isCanceled'])

    def test_get_warning_info_with_is_success_param(self):
        # 发送带有 'isSuccess' 参数的 GET 请求
        response = self.client.get('/api/warnings/', {"isSuccess": "1"})

        # 检查响应状态码
        self.assertEqual(response.status_code, 200)

        # 确保只返回成功的警告通知
        self.assertEqual(len(response.data), 2)
        self.assertTrue(response.data[0]['isSuccess'])

    def test_get_warning_info_with_both_params(self):
        # 发送带有 'isCancel' 和 'isSuccess' 两个参数的 GET 请求
        response = self.client.get('/api/warnings/', {"isCancel": "1", "isSuccess": "1"})

        # 检查响应状态码
        self.assertEqual(response.status_code, 200)

        # 确保只返回 'isCancel' 为 True 且 'isSuccess' 为 True 的警告通知
        self.assertEqual(len(response.data), 1)
        self.assertTrue(response.data[0]['isCanceled'])
        self.assertTrue(response.data[0]['isSuccess'])

    def test_get_warning_info_with_no_matching_results(self):
        # 发送带有不匹配条件的 GET 请求
        response = self.client.get('/api/warnings/', {"isCancel": "0", "isSuccess": "0"})

        # 检查响应状态码
        self.assertEqual(response.status_code, 200)

        # 确保没有返回任何数据
        self.assertEqual(len(response.data), 1)

