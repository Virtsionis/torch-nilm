import itertools
from unittest import TestCase

from torch.utils.data import DataLoader

from datasources.datasource import DatasourceFactory, NAME_UK_DALE
from datasources.torchdataset import ElectricityDataset
from modules.MyDataSet import MyChunk

APPLIANCES_UK_DALE_BUILDING_1 = ['oven', 'microwave', 'dish washer', 'fridge freezer',
                                 'kettle', 'washer dryer', 'toaster', 'boiler', 'television',
                                 'hair dryer', 'vacuum cleaner', 'light']


class TestDatasource(TestCase):

    def setUp(self) -> None:
        self.ukdale_datasource = DatasourceFactory.create_uk_dale_datasource()

    def test_electricity_dataset_iterator(self):
        datasource = DatasourceFactory.create_datasource(NAME_UK_DALE)
        train_dataset = ElectricityDataset(datasource=datasource,
                                           building=1,
                                           device="fridge",
                                           start_date="2013-05-15",
                                           end_date="2013-06-15",
                                           transform=None,
                                           window_size=50,
                                           mmax=None,
                                           sample_period=6,
                                           chunksize=50000,
                                           batch_size=32)

        train_loader = DataLoader(train_dataset, batch_size=32,
                                  shuffle=False, num_workers=4)
        count = 0
        items = set()
        doublicate = 0
        for elem in train_loader:
            count += 1
            item = tuple(elem[1].numpy())
            if item in items:
                doublicate += 1
            items.add(item)

        count2 = 0
        doublicate2 = 0
        for elem in train_loader:
            count2 += 1
            item = tuple(elem[1].numpy())
            if item in items:
                doublicate2 += 1
            items.add(item)

        print(f"dublicate {doublicate}")
        print(f"doublicate2 {doublicate2}")
        print(f"{count} == {count2}")
        self.assertEqual(count, count2)
        self.assertEqual(doublicate2, count2)


# dublicate 7700
# doublicate2 13936
# 13936 == 13936
#
# dublicate 7705
# doublicate2 13934
# 13934 == 13934