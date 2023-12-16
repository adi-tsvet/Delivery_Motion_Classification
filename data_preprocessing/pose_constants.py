BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Body_Columns = [f"{part}X" for part in BODY_PARTS.keys()] + [f"{part}Y" for part in BODY_PARTS.keys()]
# print(Body_Columns)

BODY_COLUMNS = ['NoseX', 'NoseY', 'NeckX', 'NeckY', 'RShoulderX', 'RShoulderY', 'RElbowX', 'RElbowY',
                'RWristX', 'RWristY', 'LShoulderX', 'LShoulderY', 'LElbowX', 'LElbowY', 'LWristX', 'LWristY',
                'RHipX', 'RHipY', 'RKneeX', 'RKneeY', 'RAnkleX', 'RAnkleY', 'LHipX', 'LHipY', 'LKneeX', 'LKneeY',
                'LAnkleX', 'LAnkleY', 'REyeX', 'REyeY', 'LEyeX', 'LEyeY', 'REarX', 'REarY', 'LEarX', 'LEarY',
                'BackgroundX', 'BackgroundY']


# BODY_COLUMNS_X = [f"{part}X" for part in BODY_PARTS.keys()]
# BODY_COLUMNS_Y = [f"{part}Y" for part in BODY_PARTS.keys()]
#
# # Combine X and Y columns
# BODY_COLUMNS = [x for pair in zip(BODY_COLUMNS_X, BODY_COLUMNS_Y) for x in pair]






