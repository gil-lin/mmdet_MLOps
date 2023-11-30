import os
import warnings
from argparse import Namespace
import pytz
import datetime

from allegroai import DataView, Task, OutputModel

Task.add_requirements('protobuf', '<=3.20.1')
Task.add_requirements('setuptools', '<=59.5.0')
# Task.add_requirements('albumentations', '>=0.3.2')


def init_task(args: Namespace, mode):
    """
    Initialize allegroai task object for the task.

    Args:
        args: (Namespace) arguments from parser
        mode: (str) name of mode, train/test etc'.

    Returns:
        task: (allegroai.Task) the initialized task object
        experiment_dir: (str) the path to the experiment save/load directory
    """
    # Task.debug_simulate_remote_task(task_id='c709233f4c85490ba266cddc9f3e42cd')
    experiment_dir, experiment_name = get_experiment_dir(args, mode)
    task = Task.init(
        project_name=args.project,
        task_name=experiment_name,
        auto_connect_frameworks={
            'tensorboard': False,  # automatically register tensorboard
            'pytorch': True,  # if False : don't automatically register models
            'matplotlib': True}  # don't automatically register plots
    )

    task.set_base_docker(
        docker_cmd="565683576536.dkr.ecr.eu-west-1.amazonaws.com/mmdetection:1.12.0-cuda11.3-cudnn8-devel",
        # docker_cmd="mmdetection:1.12.0-cuda11.3-cudnn8-devel",
        docker_arguments="--shm-size 8G",
    )

    # During task clone, args do not automatically update
    if experiment_name != task.name:
        assert not task.running_locally(), 'Expecting collision in task name only during remote execution'
        warnings.warn(f'New task name due to remote run - {experiment_name} != {task.name}')
        args.experiment_tag = task.name
        args.project = task.get_project_name()
        experiment_dir, experiment_name = get_experiment_dir(args, mode)

    # yaniv: If filepath is too long, this results in "OSError: [Errno 36] File name too long" later during training.
    #   A manual check showed that length of 359 gave an error. The experiment name at the time was 270 characters.
    assert len(experiment_dir) < 350, f'Experiment path is too long ({len(experiment_dir)}), ' \
                                      f'try to shorten the experiment name: "{experiment_name}"'

    return task, experiment_dir


def get_experiment_dir(args: Namespace, mode: str):
    """
    Build experiment path according to convention.

    format:
        <ROOT_DIR>/<TASK>/<PROJECT>/%YYYY-%MM-%DD--<MODE>--<EXPERIMENT_TAG>/
    example:
        <ROOT_DIR>/det/coco/2021-04-01--train--baseline/

    Args:
        args: (Namespace) arguments from parser
        mode: (str) name of mode, train/test etc'.
    """

    assert mode in ('train', 'test')
    # Get current time
    strtime = pytz.timezone('Asia/Jerusalem').localize(datetime.datetime.now()).__format__('%Y-%m-%d_%H%M%S')
    # Get experiment name
    experiment_name = f'{mode}--{args.experiment_tag or "debug"}'
    # Build path
    experiment_dir = os.path.join(args.root_dir, args.task_type, args.project,
                                  f'{strtime}--{experiment_name}')
    return experiment_dir, experiment_name


def get_dataview(
        queries,
        categories=None,
        name="default",
        iteration_order="random",
        iteration_infinite=False,
        max_num_frames=None,
        cat_mapping=None,
        special_cat_mapping=None,
        predefined_label_enums=None,
        prefetch=False,
        **kwargs,
) -> DataView:
    task = Task.current_task()
    dataviews = task.get_dataviews() if task else {}

    if name in dataviews.keys():
        print(f'Taking existing dataview [{name}]')
        dv = dataviews[name]
    else:
        print(f'Creating new dataview [{name}]')
        dv = create_dataview(
            queries=queries,
            categories=categories,
            name=name,
            iteration_order=iteration_order,
            iteration_infinite=iteration_infinite,
            max_num_frames=max_num_frames,
            cat_mapping=cat_mapping,
            special_cat_mapping=special_cat_mapping,
            predefined_label_enums=predefined_label_enums,
        )
        if task is not None:
            task.connect(dv, name=name)

    if prefetch and os.environ.get("LOCAL_RANK", '0') == '0':
        dv.prefetch_files()

    return dv


def create_dataview(
        queries,
        categories=None,
        name="default",
        iteration_order="random",
        iteration_infinite=False,
        max_num_frames=None,
        cat_mapping=None,
        special_cat_mapping=None,
        predefined_label_enums: dict = None,
        **kwargs,
) -> DataView:
    dv: DataView = DataView(
        name=name,
        iteration_order=iteration_order,
        maximum_number_of_frames=max_num_frames,
        iteration_infinite=iteration_infinite,
    )

    category_dict = {cat: i for i, cat in enumerate(categories)} if categories is not None else {}
    if predefined_label_enums is not None:
        category_dict.update(predefined_label_enums)
    dv.set_labels(**category_dict)

    for query in [queries]:
        dv.add_query(**query)

    def _add_special_cat_mapping(from_cats, to_cat):
        if from_cats is None:
            return
        for dataset_name, cats in from_cats.items():
            if cats is None:
                continue
            for cat in cats:
                if cat is None:
                    continue
                dv.add_mapping_rule(cat, to_cat, dataset_name=dataset_name, version_name="*")

    if special_cat_mapping is not None:
        for to_cat, from_cats in special_cat_mapping.items():
            _add_special_cat_mapping(from_cats, to_cat)

    if cat_mapping is not None:
        for dataset_name, mappings in cat_mapping.items():
            if mappings is None:
                continue
            for from_cat, to_cat in mappings.items():
                if from_cat is None or to_cat is None:
                    continue
                assert to_cat in category_dict, f"trying to map {from_cat} to {to_cat}, but {to_cat} " \
                                                f"is not one of the defined categories which are {category_dict}"
                dv.add_mapping_rule(from_cat, to_cat, dataset_name=dataset_name, version_name="*")

    return dv


def get_train_and_val_dataviews(
        dataview_cfg,
        categories,
        use_train_iterable_dataset: bool,
        prefetch: bool,
) -> DataView:
    dataview = get_dataview(
        name=dataview_cfg.name,
        queries=dataview_cfg.queries,
        categories=categories,
        iteration_infinite=use_train_iterable_dataset,
        iteration_order='random',
        max_num_frames=dataview_cfg.max_num_frames,
        cat_mapping=dataview_cfg.cat_mapping,
        special_cat_mapping=dataview_cfg.special_cat_mapping,
        predefined_label_enums=dataview_cfg.predefined_label_enums,
        prefetch=prefetch,
    )

    return dataview


def create_output_model(cfg, name):
    task = Task.current_task()
    if task is None:
        warnings.warn('Allegro task in not initialized. Cannot create an OutputModel')
        return None
    category_dict = {cat: i for i, cat in enumerate(cfg['config_dict']['categories'])}
    output_model = OutputModel(
        config_dict=cfg['config_dict'],
        tags=[cfg['task_type'], os.path.splitext(os.path.basename(cfg['exp_name']))[0]],
        framework='PyTorch',
        name=name,
        label_enumeration=category_dict
    )
    return output_model
