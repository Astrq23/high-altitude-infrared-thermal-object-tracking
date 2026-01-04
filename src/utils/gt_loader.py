import pandas as pd

def load_mot_gt(gt_path):
    if gt_path is None: return {}
    # Xử lý nếu gt_path là đối tượng file của Gradio hoặc string
    if hasattr(gt_path, 'name'): 
        gt_path = gt_path.name
        
    gt_dict = {}
    try:
        try:
            df = pd.read_csv(gt_path, header=None, sep=',')
        except:
            df = pd.read_csv(gt_path, header=None, sep=r'\s+', engine='python')
            
        for index, row in df.iterrows():
            try:
                if not str(row[0]).replace('.','',1).isdigit(): continue
                frame = int(float(row[0]))
                obj_id = int(float(row[1]))
                x1, y1, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                
                if frame not in gt_dict:
                    gt_dict[frame] = []
                gt_dict[frame].append([x1, y1, x1+w, y1+h, obj_id])
            except ValueError:
                continue
    except Exception as e:
        print(f"GT Error: {e}")
        return {}
    return gt_dict