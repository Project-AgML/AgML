import os
import abc

from agml.utils.downloads import download_dataset  # noqa
from agml.backend.config import default_data_save_path
from agml.backend.tftorch import _swap_loader_mro
from agml.data.metadata import DatasetMetadata

class AgMLDataLoader(object):
    """Loads and processes a dataset from the AgML public repository.

    An `AgMLDataLoader` contains data from one public data source in
    the AgML public repository, and exposes methods allowing easy
    usage both within an AgML pipeline and within an existing pipeline
    in other frameworks like TensorFlow and PyTorch.

    The class should be instantiated with one dataset name, which
    will in turn be downloaded (and simply loaded in subsequent runs).
    Images and annotations can be accessed from relevant properties and
    methods, and transforms from multiple backends can be applied.

    The actual data content within the dataset can be exported in one
    of two ways: using `export_contents()` to get a dictionary with the
    image paths and relevant annotations (too expensive to pre-load in
    all of the images), or using one of the specific methods designed
    for backend integration, namely `torch()` to get a pre-instantiated
    `torch.utils.data.DataLoader` with the dataset pre-loaded in, or
    `tensorflow()` to get a `tf.data.Dataset` with the dataset loaded
    in. See the relevant guides and the method documentations for
    additional info on how to use these methods.

    Notes
    -----
    See the derived classes `AgMLImageClassificationDataLoader`,
    `AgMLObjectDetectionDataLoader`, and `AgMLSemanticSegmentationDataLoader`
    for information on the specific functionalities of the data loader
    for the different types of ML tasks supported in AgML.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
    kwargs : Any
        Other initialization parameters. Can include:
        dataset_path : str
            The local path to the dataset.
        overwrite : str
            Whether to re-download an existing dataset.
    """

    def __new__(cls, dataset, **kwargs):
        if kwargs.get('skip_init', False):
            return super(AgMLDataLoader, cls).__new__(cls)
        ml_task_type = DatasetMetadata(dataset).tasks.ml
        if ml_task_type == 'image_classification':
            from agml.data.classification \
                import AgMLImageClassificationDataLoader
            return AgMLImageClassificationDataLoader(
                dataset, skip_init = True, **kwargs)
        elif ml_task_type == 'semantic_segmentation':
            from agml.data.segmentation \
                import AgMLSemanticSegmentationDataLoader
            return AgMLSemanticSegmentationDataLoader(
                dataset, skip_init = True, **kwargs)
        elif ml_task_type == 'object_detection':
            from agml.data.detection \
                import AgMLObjectDetectionDataLoader
            return AgMLObjectDetectionDataLoader(
                dataset, skip_init = True, **kwargs)
        raise ValueError(
            f"Unsupported ML task: {ml_task_type}. Please report this error.")

    def __init__(self, dataset, **kwargs):
        # Setup and find the dataset.
        self._info = DatasetMetadata(dataset)
        self._find_dataset(**kwargs)

        # Process internal logic.
        self._shuffle = True
        if not kwargs.get('shuffle', True):
            self._shuffle = False
        self._default_image_size = (512, 512)
        if not kwargs.get('image_size', True):
            self._default_image_size = kwargs['image_size']

        # Parameters that may or may not be changed. If not, they are
        # here just for consistency in the class and internal methods.
        self._is_batched = False
        self._batched_data = None

        self._preprocessing_enabled = True

        self._training_data = None
        self._validation_data = None
        self._test_data = None

        self._getitem_as_batch = False

    @abc.abstractmethod
    def __len__(self):
        """Returns the number of images in the dataset."""
        if self._is_batched:
            return len(self._batched_data)
        return self.num_images

    def __str__(self):
        fmt = (f"<AgMLDataLoader [{self._info.name}] "
               f"[task = {self._info.tasks.ml}]")
        if hasattr(self, '_split_name'):
            fmt += f" [split = {getattr(self, '_split_name')}]"
        fmt += ">"
        return fmt

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def _find_dataset(self, **kwargs):
        """Searches for or downloads the dataset in this loader."""
        self._stored_kwargs_for_init = kwargs
        if not kwargs.get('overwrite', False):
            if kwargs.get('dataset_path', False):
                self._dataset_root = kwargs['dataset_path']
                if not os.path.exists(self._dataset_root):
                    raise FileNotFoundError(
                        f"Provided dataset path does not "
                        f"exist: {self._dataset_root}.")
                return
            elif os.path.exists(os.path.join(
                    default_data_save_path(), self._info.name)):
                self._dataset_root = os.path.join(
                    default_data_save_path(), self._info.name)
                return
        if kwargs.get('dataset_download_path', False):
            download_path = kwargs['dataset_download_path']
            if not os.path.exists(download_path):
                raise NotADirectoryError(
                    f"Invalid dataset download path: {download_path}.")
        else:
            download_path = default_data_save_path()
        print(f"[AgML Download]: Downloading dataset `{self._info.name}` "
              f"to {os.path.join(download_path, self._info.name)}.")
        download_dataset(self._info.name, download_path)
        self._dataset_root = os.path.join(
            default_data_save_path(), self._info.name)

    @property
    def name(self):
        return self._info.name

    @property
    def num_images(self):
        return self._info.num_images

    @property
    def dataset_root(self):
        # The local path to the dataset.
        return self._dataset_root

    @property
    def info(self):
        """Contains metadata information about the dataset.

        This property returns a `DatasetMetadata` object which
        contains a number of properties with information about
        the dataset. See `DatasetMetadata` for more information.
        """
        return self._info

    def shuffle(self):
        """Shuffles the data in this loader."""
        self._reshuffle()

    def disable_preprocessing(self):
        """Disables internal preprocessing for `__getitem__`.

        By default, images loaded from this dataset when using `__getitem__`
        will apply a transformation or general preprocessing pipeline if
        provided. This method disables these pipelines and returns just the
        original image and its label from the `__getitem__` method.

        To re-enable preprocessing, run `enable_preprocessing()`.
        """
        self._preprocessing_enabled = False

    def enable_preprocessing(self):
        """Re-enables internal processing for `__getitem__`.

        If the preprocessing pipeline in the dataset is disabled using
        `disable_preprocessing()`, this method re-enables preprocessing.
        """
        self._preprocessing_enabled = True

    @property
    def training_data(self):
        if self._training_data is not None:
            ret_cls = self.__class__._from_extant_data(
                self._wrap_reduced_data('training'),
                self._stored_kwargs_for_init)
            ret_cls._split_name = 'train'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'train' "
            "parameter to use the `training_data` property.")

    @property
    def validation_data(self):
        if self._validation_data is not None:
            ret_cls = self.__class__._from_extant_data(
                self._wrap_reduced_data('validation'),
                self._stored_kwargs_for_init)
            ret_cls._split_name = 'validation'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'val' "
            "parameter to use the `validation_data` property.")

    @property
    def test_data(self):
        if self._test_data is not None:
            ret_cls = self.__class__._from_extant_data(
                self._wrap_reduced_data('test'),
                self._stored_kwargs_for_init)
            ret_cls._split_name = 'test'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'test' "
            "parameter to use the `test_data` property.")

    def as_keras_sequence(self):
        _swap_loader_mro(self, 'tf')
        self._getitem_as_batch = True

    def as_torch_dataset(self):
        _swap_loader_mro(self, 'torch')
        self._getitem_as_batch = True

    @abc.abstractmethod
    def _wrap_reduced_data(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _from_extant_data(cls, *args, **kwargs):
        raise NotImplementedError()

    #### API METHODS - OVERWRITTEN BY DERIVED CLASSES ####

    @abc.abstractmethod
    def _reshuffle(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def batch(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def export_contents(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def torch(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def tensorflow(self, *args, **kwargs):
        raise NotImplementedError()




