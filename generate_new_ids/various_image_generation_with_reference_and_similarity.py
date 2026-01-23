# on generate_new_ids
# python various_image_generation_with_reference_and_similarity.py --image_file /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS --model_weights ../weights/vec2face_generator.pth --batch_size 5 --example 50 --name images-of-references --similarity-range [0.5,0.69] --num-new-ids 3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import argparse
import pixel_generator.vec2face.model_vec2face as model_vec2face
import imageio
from tqdm import tqdm
import torch.optim as optim
import numpy as np
# from glob import glob    # original
import glob                # Bernardo
import os

from models import iresnet
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import re
import json
import time



class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB').resize((112, 112))
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img, img_path

def get_args_parser():
    parser = argparse.ArgumentParser('Vec2Face verify', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='vec2face_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--image_file', type=str, default='/hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS',
                        help='file path with images')
    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')

    # Pre-trained enc parameters
    parser.add_argument('--use_rep', action='store_false', help='use representation as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='use class label as condition.')
    parser.add_argument('--rep_dim', default=512, type=int)

    # Pixel generation parameters
    parser.add_argument('--rep_drop_prob', default=0.0, type=float)
    parser.add_argument('--feat_batch_size', default=2000, type=int)
    parser.add_argument('--example', default=50, type=int)
    parser.add_argument('--name', default=None, type=str)

    # Vec2Face params
    parser.add_argument('--mask_ratio_min', type=float, default=0.1,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.15,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--model_weights', default='',
                        help='model weights')

    def parse_list_arg(arg_string):
        try:
            values = [float(item.strip().strip('[').strip(']')) for item in arg_string.split(',')]
            return values
        except ValueError:
            raise argparse.ArgumentTypeError("List values must be floats separated by commas, e.g., '0.5,0.69'")

    parser.add_argument("--path-subj-list",    type=str, default="")   # /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/merge_with_dataset_MS-Celeb-1M-ms1m-retinaface-t1-imgs_FACE_EMBEDDINGS_sim-range=[0.5,0.69]/dict_paths_new_subjs_base_subjs.json
    parser.add_argument("--similarity-range",  type=parse_list_arg, default=[0.5,0.69], required=True, help='A list of float values separated by commas, e.g., 0.5,0.69 or [0.5,0.69]')
    parser.add_argument("--num-new-ids",       type=int, default=-1)   # -1 == one new synthetic id for each real id
    # parser.add_argument("--num-samples-by-id", type=int, default=50)
    parser.add_argument("--path-output",       type=str, default="")

    return parser.parse_args()


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    # print(f'Found files: {len(file_list)}', end='\r')
    # print()
    file_list = natural_sort(file_list)
    return file_list


def get_immediate_subdirs(parent_dir=''):
    subdirs = [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    subdirs = natural_sort(subdirs)
    return subdirs


def load_json(path_file=''):
    assert os.path.isfile(path_file), f"Error, no such file: \'{path_file}\'"
    with open(path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(dict_data={}, path_file=''):
    dir_name = os.path.dirname(path_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, indent=4, ensure_ascii=False)


def save_images(images, id_num, root, name):
    # global j, prev_id
    # save_root = f"{root}/{name}"
    for j, image in enumerate(images):
        # save_folder = f"{root}/{id_num[j]}/"
        # save_folder = f"{save_root}"
        os.makedirs(root, exist_ok=True)
        # if prev_id != id_num[i]:
        #     prev_id = id_num[i]
        #     j = 0
        imageio.imwrite(f"{root}/{str(id_num[j]).zfill(3)}.jpg", image)
        # j += 1


def sample_nearby_vectors(base_vector, epsilons, percentages=[0.4, 0.4, 0.2]):
    row, col = base_vector.shape
    norm = torch.norm(base_vector, 2, 1, True)
    diff = []
    for i, eps in enumerate(epsilons):
        diff.append(np.random.normal(0, eps, (int(row * percentages[i]), col)))
    diff = np.vstack(diff)
    np.random.shuffle(diff)
    # diff = torch.tensor(diff)           # original
    diff = torch.tensor(diff).to(device)  # Bernardo
    # print(diff.shape)
    generated_samples = base_vector + diff
    generated_samples = generated_samples / torch.norm(generated_samples, 2, 1, True) * norm
    return generated_samples


def _create_fr_model(model_path="../weights/magface-r100-glint360k.pth", depth="100"):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def processing_images(file_path, feature_model):
    if os.path.isdir(file_path):
        # ref_images = glob(f"{file_path}/*")                   # original
        ref_images = glob.glob(f"{glob.escape(file_path)}/*")   # Bernardo
    elif os.path.isfile(file_path):
        # ref_images = np.genfromtxt(file_path, str)            # original
        ref_images = [file_path]                                # Bernardo
    else:
        raise AttributeError("Please give either a folder path of images or a file path of images.")
    
    dataset = ImageDataset(ref_images)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    feature_model = feature_model.to(device)
    feature_model.eval()

    features = []
    im_ids = []

    with torch.no_grad():
        # for batch, paths in tqdm(dataloader, desc="Processing images"):
        for batch, paths in dataloader:
            batch = batch.to(device)
            batch_features = feature_model(batch)
            features.append(batch_features.cpu())
            im_ids.extend([path.split("/")[-1][:-4] for path in paths])

    features = torch.cat(features, dim=0)
    return features, im_ids


def get_random_float(min_max_list):
    if len(min_max_list) == 1:
        min_max_list.append(min_max_list[0])
    min_val, max_val = min_max_list
    factor = 100
    
    scaled_min = round(min_val * factor)
    scaled_max = round(max_val * factor)
    
    random_int = random.randint(scaled_min, scaled_max)
    random_float = random_int / factor
    return random_float


def load_embedding(embedd_path=''):
    if embedd_path.endswith('.pt'):
        embedd = torch.load(embedd_path).detach()
        embedd = torch.squeeze(embedd)
    elif embedd_path.endswith('.npy'):
        embedd = np.load(embedd_path)
        embedd = np.squeeze(embedd)
    else:
        raise Exception(f'File format not supported: \'{embedd_path}\'')
    return embedd


def rotate_embedding_by_cosine_similarity(v1: torch.Tensor, cosine_similarity: float) -> torch.Tensor:
    v1 = torch.squeeze(v1)
    if not (0.0 <= cosine_similarity <= 1.0):
        raise ValueError("Cosine similarity must be between 0.0 and 1.0.")
    if v1.dim() != 1 or v1.size(0) != 512:
        raise ValueError("Input tensor must be 512-dimensional (1D tensor).")
    
    theta = torch.acos(torch.tensor(cosine_similarity, device=v1.device))
    
    if torch.isclose(theta, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
    
    v1_norm = torch.linalg.norm(v1).to(torch.float32)
    if torch.isclose(v1_norm, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
        
    u1 = v1 / v1_norm
    random_vector = torch.randn_like(v1)
    projection_onto_u1 = torch.dot(random_vector, u1) * u1
    u2_raw = random_vector - projection_onto_u1
    u2_norm = torch.linalg.norm(u2_raw).to(torch.float32)
    
    if torch.isclose(u2_norm, torch.tensor(0.0).to(torch.float32)):
        raise RuntimeError("Failed to generate a non-collinear random vector. Try running again.")

    u2 = u2_raw / u2_norm
    u1_prime = (u1 * torch.cos(theta)) + (u2 * torch.sin(theta))
    v1_prime = u1_prime * v1_norm
    v1_prime = torch.unsqueeze(v1_prime, 0)
    return v1_prime





if __name__ == '__main__':
    args = get_args_parser()
    j = 0
    prev_id = -1

    dim = args.rep_dim
    batch_size = args.feat_batch_size
    example = args.example
    name = args.name
    image_path = args.image_file
    input_size = args.input_size



    if not args.path_output:
        args.path_output = f"{args.image_file}_newSynthIDs_Vec2Face_sim={args.similarity_range}".replace(' ','')
    else:
        args.path_output = os.path.join(args.path_output, f"{args.image_file.split('/')[-1]}_newSynthIDs_Arc2Face_sim={args.similarity_range}".replace(' ',''))



    print("Loading model...")
    device = torch.device('cuda')
    model = model_vec2face.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                                mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                                use_rep=args.use_rep,
                                                rep_dim=args.rep_dim,
                                                rep_drop_prob=args.rep_drop_prob,
                                                use_class_label=args.use_class_label)

    model = model.to(device)
    checkpoint = torch.load(args.model_weights, map_location='cuda')
    model.load_state_dict(checkpoint['model_vec2face'])
    model.eval()

    print("Loading estimators...")
    # quality model
    scorer = _create_fr_model().to(device)
    # id model
    fr_model = _create_fr_model("../weights/arcface-r100-glint360k.pth").to(device)



    if args.path_subj_list:
        json_subjs_list = load_json(args.path_subj_list)
        subjs_orig_paths = [subj_path[0][0] for subj_path in json_subjs_list.values()]
        subjs_names = [subj_name.split('/')[-2] for subj_name in subjs_orig_paths]
    else:
        subjs_orig_paths = get_immediate_subdirs(args.image_file)
        subjs_names = [subj_name.split('/')[-1] for subj_name in subjs_orig_paths]

    if args.num_new_ids == 0:
        print('\n--num-new-ids == 0, no new synthetic identities will be generated!')
        sys.exit(0)
    elif args.num_new_ids > 0:
        # random.seed(440)
        random.shuffle(subjs_names)
        subjs_names = subjs_names[:args.num_new_ids]
        args.path_output += f'_{args.num_new_ids}ids'


    for idx_subj, subj_name in enumerate(subjs_names):
        start_time = time.time()
        print(f'id {idx_subj}/{len(subjs_names)} - subj_name {subj_name}')
        path_dir_subj = os.path.join(args.image_file, subj_name)
        path_mean_embedding_subj = get_all_files_in_path(path_dir_subj, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern='_mean_embedding_')

        if len(path_mean_embedding_subj) > 0:
            print('    Loading identity mean embedding')
            src_id_emb = load_embedding(path_mean_embedding_subj[0])
        else:
            paths_files = get_all_files_in_path(path_dir_subj, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'])
            src_id_embedds = torch.zeros((len(paths_files), 512))
            for idx_path, path_file in enumerate(paths_files):
                if path_file.endswith('.npy') or path_file.endswith('.pt'):
                    src_id_embedds[idx_path] = load_embedding(path_file)
                else:
                    img = np.array(Image.open(path_file))[:,:,::-1]
                    if img.shape[0] == 112 and img.shape[1] == 112:   # face already aligned
                        # reference_ids, im_ids = processing_images(path_file, fr_model)
                        src_id_embedds[idx_path], _ = processing_images(path_file, fr_model)
                        # im_ids = [item for item in im_ids for _ in range(example)]
                    else:
                        faces = fr_model.get(img)   # detect face
                        if len(faces) == 0:   # no face detected
                            raise Exception(f'No face detected in image: \'{path_file}\'')
                        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                        src_id_embedds[idx_path] = faces['embedding']
            src_id_emb = src_id_embedds.mean(axis=0)
        
        src_id_emb = torch.tensor(src_id_emb, dtype=torch.float32)[None].cuda()

        similarity = get_random_float(args.similarity_range)
        print('    similarity:', similarity)

        # Generate new identity embedding
        new_id_emb = rotate_embedding_by_cosine_similarity(src_id_emb, similarity)
        # new_id_emb = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding



        expanded_ids = torch.repeat_interleave(new_id_emb, example, dim=0).to(torch.float32)
        samples = sample_nearby_vectors(expanded_ids,
                                        epsilons=[0.2],
                                        percentages=[1.]).to(torch.float32)
        samples = samples.to(device, non_blocking=True)

        collection = []
        im_ids = list(range(args.example))
        print("Generating and saving images...")
        for i in tqdm(range(0, len(expanded_ids), args.batch_size)):
            im_features = samples[i: i + args.batch_size]

            # image, _ = model.gen_image(im_features, scorer, fr_model, class_rep=im_features,
            #                            q_target=27)

            _, _, image, *_ = model(im_features)  # for faster processing, but no guarantee for quality
            # print('Saving images...')
            save_images(((image.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8),
                        im_ids[i: i + args.batch_size],
                        f"various_generated_images_ref/{subj_name}_{similarity}",
                        name)

        exec_time = time.time() - start_time
        remain_time = exec_time * (len(subjs_names)-idx_subj+1)
        print('    Exec time: %.2fsec    %.2fmin    %.2fhour' % (exec_time, exec_time/60, exec_time/3600))
        print('    Remaining time: %.2fsec    %.2fmin    %.2fhour' % (remain_time, remain_time/60, remain_time/3600))
        print('------------')

    print('\nFinished!')
