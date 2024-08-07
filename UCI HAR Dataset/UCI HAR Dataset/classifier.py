import json
from sensiml.dclproj import DataSegments, DataSegment
import numpy as  np

def to_data_studio(datasegments: DataSegments):
    
    result = []
    for datasegment in datasegments:
        tmp = {'SegmentStart': datasegment.start,
               'SegmentEnd': datasegment.end,
               'ClassificationName':datasegment.label_value}
        
        result.append(tmp)
        
    return result
 
 
def validate_params(input_contract, params: dict) -> bool:
    
    params = json.loads(params)
        
    for ic_param in input_contract:
        found = False
        if ic_param['name'] in params:
            found=True
            
        if not found:
            raise Exception(f"param {ic_param['name']} is required")
        
    

    return params

def dict_to_datasegments(input_dict: dict, dtype=np.int32):
    columns = list(input_dict.keys())

    data = np.vstack(
        [np.array(input_dict[column], dtype=dtype).T for column in columns]
    )
    

    return DataSegments([DataSegment(data=data, columns=columns, segment_id=0, capture_sample_sequence_start=0, capture_sample_sequence_end=data.shape[0]-1)])


def convert_to_datasegments(data) -> DataSegments:

    converted_data = {}
    for k, v in dict(data).items():
        converted_data[k] = [int(item) for item in v]

    return dict_to_datasegments(converted_data)
 
def get_info_json() -> str:
    return json.dumps(get_info())
    
def get_info() -> dict:
    return {
        "name": "Random Classifier",
        "type": "Python",
        "subtype": "Segmentation and Classification",
        "description": "This algorithm will perform a sliding window, then assign a class of either A or B to the model",
        "input_contract": [
            {
                "name": "columns",
                "type": "List(str)",                            
            },
            {
                "name": "window_size",
                "type": "int",
                "default": 400,
                "range": [200, 1000],
            },
            {
                "name": "delta",
                "type": "int",
                "default": 400,
                "range": [200, 1000],
            },             
        ],
        "output_contract": [],
    }

def classify(datasegments: DataSegments, params: dict) -> DataSegments:

    last = 0
    for datasegment in datasegments:
        col_index = datasegment.get_column_index(params['columns'][0])
        f1 = datasegment.data[col_index, :].sum() 
        datasegment.feature_vector = [f1]
            
        if datasegment.feature_vector[0] > last:
            datasegment.label_value = "A"
        else:
            datasegment.label_value = "B"

        last = datasegment.feature_vector[0]

    return datasegments



def recognize_capture(data, params):    
    data_segments = convert_to_datasegments(data)
    
    params = validate_params(get_info()['input_contract'], params)
    
    data_segments = segment_data(data_segments, params['window_size'], params['delta'])
        
    data_segments = classify(data_segments, params)
    

    return to_data_studio(data_segments)

def segment_data(
    input_data: DataSegments, window_size: int, delta: int
) -> DataSegments:
    new_segments = []
    for segment in input_data:
        for segment_id, start_index in enumerate(
            range(0, segment.data.shape[1] - (window_size - 1), delta)
        ):
            tmp_segment = DataSegment(
                segment_id=segment_id,
                columns=segment.columns,
                capture_sample_sequence_start=start_index,
                capture_sample_sequence_end=start_index + window_size,
            )
            tmp_segment._data = segment.data[
                :, start_index : start_index + window_size
            ]
            
            tmp_segment.label_value = "Unknown"
            
            new_segments.append(tmp_segment)

    return DataSegments(new_segments)