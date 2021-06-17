from unittest import TestCase

from datasources.datasource import DatasourceFactory

APPLIANCES_UK_DALE_BUILDING_1 = ['oven', 'microwave', 'dish washer', 'fridge freezer',
                                 'kettle', 'washer dryer', 'toaster', 'boiler', 'television',
                                 'hair dryer', 'vacuum cleaner', 'light']


class TestDatasource(TestCase):

    def setUp(self) -> None:
        self.ukdale_datasource = DatasourceFactory.create_uk_dale_datasource()

    def test_get_mains_generator(self):
        chunksize = 5000
        mains_generator = self.ukdale_datasource.get_mains_generator("2014-09-01", "2014-09-10", chunksize=chunksize)
        counter = 0
        for item in mains_generator:
            if len(item) != chunksize:
                counter += 1
            if len(item) > chunksize:
                self.fail()
        self.assertLessEqual(counter, 1)

    def test_get_appliance_generator(self):
        chunksize = 5000
        appliance_generator = self.ukdale_datasource.get_appliance_generator("oven",
                                                                             "2014-09-01", "2014-09-10",
                                                                             chunksize=chunksize)
        counter = 0
        for item in appliance_generator:
            if len(item) != chunksize:
                counter += 1
            if len(item) > chunksize:
                self.fail()
        self.assertLessEqual(counter, 1)
