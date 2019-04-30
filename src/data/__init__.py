from importlib import import_module
from dataloader import MSDataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
import pdb
# from m_dataset import SRdataset, SRDataset_HDF5Loader

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu
            ))


# class Data_hdf5:
#     def __init__(self, args):
#         self.loader_train = None
#         if not args.test_only:
#             hdf5_train_files = os.listdir(args.h5_train_files)
#             train_data = SRdataset(hdf5_train_files)
#             self.loader_train = DataLoader(dataset = train_data, batch_size = args.batch_size)            

#         self.loader_test = []
#         hdf5_test_files = os.listdir(args.h5_test_files)
#         test_data = SRdataset(hdf5_test_files)
#         self.loader_test = DataLoader(dataset = test_data, batch_size = 1)

