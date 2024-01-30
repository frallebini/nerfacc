from .nerf_synthetic import SubjectLoader


CLASS_DICT = {
        # https://github.com/Xharlie/ShapenetRender_more_variation?tab=readme-ov-file#dataset-intro
        "02691156": 0,  # airplane
        "02828884": 1,  # bench
        "02933112": 2,  # cabinet
        "02958343": 3,  # car
        "03001627": 4,  # chair
        "03211117": 5,  # display
        "03636649": 6,  # lamp
        "03691459": 7,  # speaker
        "04090263": 8,  # rifle
        "04256520": 9,  # sofa
        "04379243": 10, # table
        "04401088": 11, # phone
        "04530566": 12  # watercraft
    }

TO_SKIP = [
    "02691156/f6373cc88634e8ddaf781741e31f0df4_A1", 
    "04090263/60a861a5b416030a93153dd7e0ee121c_A2"
]

class ShapeNetLoader(SubjectLoader):

    SPLITS = ["train", "test"]
    WIDTH, HEIGHT = 224, 224

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "random",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__(
            subject_id, 
            root_fp, 
            split, 
            color_bkgd_aug, 
            num_rays,
            near,
            far,
            batch_over_images,
            device
        )
