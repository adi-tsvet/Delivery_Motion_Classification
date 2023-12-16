import pandas as pd
import numpy as np
from data_preprocessing.pose_constants import BODY_COLUMNS, BODY_PARTS


def calculate_displacement(csv_path):
    # Read CSV data into numpy arrays
    df = pd.read_csv(csv_path)
    data_array = df.to_numpy()

    # Initialize displacement_data to store the displacement of each body part between frames
    displacement_data = {body_part: [] for body_part in BODY_PARTS.keys()}

    # Calculate displacement for each body part between frames using a reference point in each frame
    for i in range(1, len(data_array)):
        reference_point_x = (data_array[i, BODY_COLUMNS.index('RHipX')] + data_array[
            i, BODY_COLUMNS.index('LHipX')]) / 2
        reference_point_y = (data_array[i, BODY_COLUMNS.index('RHipY')] + data_array[
            i, BODY_COLUMNS.index('LHipY')]) / 2

        for body_part in BODY_PARTS.keys():
            # Calculate displacement between current and previous frame for each body part
            x_current = data_array[i, BODY_COLUMNS.index(body_part + 'X')]
            y_current = data_array[i, BODY_COLUMNS.index(body_part + 'Y')]

            displacement = np.linalg.norm([x_current - reference_point_x, y_current - reference_point_y])
            displacement_data[body_part].append(displacement)

    # Convert displacement_data to a DataFrame
    df_displacement = pd.DataFrame(displacement_data)

    return df_displacement