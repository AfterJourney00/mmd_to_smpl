mmd_skeleton = [
    "センター",
    "足.L",
    "ひざ.L",
    "足首.L",
    "つま先.L",
    "足.R",
    "ひざ.R",
    "足首.R",
    "つま先.R",
    "上半身",
    "上半身1",
    "上半身2",
    "首",
    "頭",
    "肩.L",
    "腕.L",
    "ひじ.L",
    "手首.L",
    "人指１.L",
    "人指２.L",
    "人指３.L",
    "中指１.L",
    "中指２.L",
    "中指３.L",
    "小指１.L",
    "小指２.L",
    "小指３.L",
    "薬指１.L",
    "薬指２.L",
    "薬指３.L",
    "親指０.L",
    "親指１.L",
    "親指２.L",
    "肩.R",
    "腕.R",
    "ひじ.R",
    "手首.R",
    "人指１.R",
    "人指２.R",
    "人指３.R",
    "中指１.R",
    "中指２.R",
    "中指３.R",
    "小指１.R",
    "小指２.R",
    "小指３.R",
    "薬指１.R",
    "薬指２.R",
    "薬指３.R",
    "親指０.R",
    "親指１.R",
    "親指２.R"
]

smpl_skeleton = [
    0,
    1,
    4,
    7,
    10,
    2,
    5,
    8,
    11,
    3,
    6,
    9,
    12,
    15,
    13,
    16,
    18,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    14,
    17,
    19,
    21,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51
]

hand_joint_idx = [i for i in range(18, 33)] + [i for i in range(37, 52)]

MMD_BONE_MAPPING = {    # record the correspondence of bone tail
    "spine1":          ("上半身.head", "上半身.tail"),
    "neck":            ("首.head", "首.tail"),          # spine3 -> neck
    "head":            ("頭.head", "頭.tail"),          # neck -> head
    "left_collar":     ("首.head", "肩.L.head"),        # spine3 -> left_collar
    "right_collar":    ("首.head", "肩.R.head"),        # spine3 -> right_collar
    "left_shoulder":   ("肩.L.head", "肩.L.tail"),      # left_collar -> left_shoulder
    "right_shoulder":  ("肩.R.head", "肩.R.tail"),      # right_collar -> right_shoulder
    "left_elbow":      ("腕.L.head", "腕.L.tail"),      # left_shoulder -> left_elbow
    "right_elbow":     ("腕.R.head", "腕.R.tail"),      # right_shoulder -> right_elbow
    "left_wrist":      ("ひじ.L.head", "ひじ.L.tail"),  # left_elbow -> left_wrist
    "right_wrist":     ("ひじ.R.head", "ひじ.R.tail"),  # right_elbow -> right_wrist
    "left_hip":        ("上半身.head", "足.L.head"),    # pelvis -> left_hip
    "right_hip":       ("上半身.head", "足.R.head"),    # pelvis -> right_hip
    "left_knee":       ("足.L.head", "足.L.tail"),      # left_hip -> left_knee
    "right_knee":      ("足.R.head", "足.R.tail"),      # right_hip -> right_knee
    "left_ankle":      ("ひざ.L.head", "ひざ.L.tail"),  # left_knee -> left_ankle
    "right_ankle":     ("ひざ.R.head", "ひざ.R.tail"),  # right_knee -> right_ankle
    "left_foot":       ("足首.L.head", "足首.L.tail"),  # left_ankle -> left_foot
    "right_foot":      ("足首.R.head", "足首.R.tail"),  # right_ankle -> right_foot
}

SMPL_BONE_JOINTS = {
    'spine1':          (0, 3),    # pelvis -> spine1
    # "spine2":          (3, 6),  # spine1 -> spine2
    # "spine3":          (6, 9),  # spine2 -> spine3
    "neck":            (9, 12),   # spine3 -> neck
    "head":            (12, 15),  # neck -> head
    "left_collar":     (9, 13),   # spine3 -> left_collar
    "right_collar":    (9, 14),   # spine3 -> right_collar
    "left_shoulder":   (13, 16),  # left_collar -> left_shoulder
    "right_shoulder":  (14, 17),  # right_collar -> right_shoulder
    "left_elbow":      (16, 18),  # left_shoulder -> left_elbow
    "right_elbow":     (17, 19),  # right_shoulder -> right_elbow
    "left_wrist":      (18, 20),  # left_elbow -> left_wrist
    "right_wrist":     (19, 21),  # right_elbow -> right_wrist
    "left_hip":        (0, 1),    # pelvis -> left_hip
    "right_hip":       (0, 2),    # pelvis -> right_hip
    "left_knee":       (1, 4),    # left_hip -> left_knee
    "right_knee":      (2, 5),    # right_hip -> right_knee
    "left_ankle":      (4, 7),    # left_knee -> left_ankle
    "right_ankle":     (5, 8),    # right_knee -> right_ankle
    "left_foot":       (7, 10),   # left_ankle -> left_foot
    "right_foot":      (8, 11),   # right_ankle -> right_foot
}

MARKER_SYMMETRIC_PAIRS = [
    ('left_chest', 'right_chest'),
    ('left_waist', 'right_waist'),
    ('left_back', 'right_back'),
    ('left_down_back', 'right_down_back'),
    ('front_left_head', 'front_right_head'),
    ('back_left_head', 'back_right_head'),
    ('top_left_arm', 'top_right_arm'),
    ('outter_left_elbow', 'outter_right_elbow'),
    ('inner_left_elbow', 'inner_right_elbow'),
    ('lower_left_arm', 'lower_right_arm'),
    ('front_left_wrist', 'front_right_wrist'),
    ('back_left_wrist', 'back_right_wrist'),
    ('left_upper_leg', 'right_upper_leg'),
    ('left_outter_knee', 'right_outter_knee'),
    ('left_inner_knee', 'right_inner_knee'),
    ('left_lower_leg', 'right_lower_leg'),
    ('left_ankle', 'right_ankle'),
    ('left_front_foot', 'right_front_foot'),
    ('left_back_foot', 'right_back_foot'),
    ('inner_top_left_leg', 'inner_top_right_leg'),
]