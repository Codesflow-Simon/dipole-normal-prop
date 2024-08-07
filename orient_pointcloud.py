import torch
from options import get_parser
import options
import util
from util import orient_center
from inference_utils import load_model_from_file, fix_n_filter, voting_policy
import field_utils
torch.manual_seed(1)


def run(opts):
    MyTimer = util.timer_factory()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    pc = util.xyz2tensor(open(opts.pc, 'r').read()).to(device)
    if opts.estimate_normals:
        with MyTimer('estimating normals'):
            # STEP: estimate normals, this just uses open3d to estimate normals
            pc = util.estimate_normals(pc, max_nn=opts.n)
    # STEP: apply transformation, this is just centering and scaling the point cloud
    pc, transform = util.Transform.trans(pc)
    input_pc = pc.clone()

    # Define some layers and models
    softmax = torch.nn.Softmax(dim=-1) # Comes in a bit later
    n_models = len(opts.models)
    models = [load_model_from_file(opts.models[i], device) for i in range(n_models)]

    with MyTimer('divide patches'):
        # STEP: divide patches, seems to just use a square grid to divide the point cloud
        patch_indices = util.divide_pc(input_pc[:, :3], opts.number_parts,
                                                                 min_patch=opts.minimum_points_per_patch)
        all_patches_indices = [x.clone() for x in patch_indices]

    with MyTimer('filter patches'):
        # STEP: filter patches, does some filtering based on curvature (eigenvalues of covariance matrix)
        patch_indices = fix_n_filter(input_pc, patch_indices, opts.curvature_threshold)


    num_patches = len(patch_indices)
    print(f'number of patches {num_patches}')

    with MyTimer('orient center'):
        for i, p in patch_indices:
            # STEP: orient center, does some global flipping of normals
            input_pc[p] = orient_center(input_pc[p]) # is this just global orientation? No, that is below

    pc_probs = torch.ones_like(input_pc[:, 0])

    for iter in range(opts.iters): # Global iterations
        with MyTimer(f'iteration {iter}'):
            [model.to(device) for model in models]
            for i, (pindx, points_indices) in enumerate(patch_indices): # Iterate patches
                with torch.no_grad():
                    data = input_pc[points_indices] # Get patch
                    data = data.to(device)
                    votes = [model(data.clone()) for model in models] # Get each model (PointCNN) output
                    # The model outputs logits in shape (N, 2), we take the second column as the probability of being positive?
                    vote_probabilities = [softmax(scores)[:, 1] for scores in votes] # Softmax the logits
                    flip, probs = voting_policy(vote_probabilities) # Each model votes for a flip, this is just an average, flip is prob<0.5
                    probs[flip] = 1 - probs[flip] # At flip indicies, invert the probability
                    pc_probs[points_indices] = probs # Set the probability of the patch to the average of the models

                    # Here is our betas!!!


                    input_pc[points_indices[flip], 3:] *= -1 # Flip the normals

            if iter % opts.propagation_iters == 0 and (iter != 0 or opts.propagation_iters == 1):
                [model.to('cpu') for model in models]
                with torch.no_grad():
                    with MyTimer('propagation'):
                        # STEP: propagate, is this alligning patches together?
                        field_utils.strongest_field_propagation(input_pc, patch_indices, all_patches_indices,
                                                                diffuse=opts.diffuse,
                                                                weights=pc_probs if opts.weighted_prop else None)

    with MyTimer('propagation'):
        # STEP: propagate, select best solution from avaliable models?
        field_utils.strongest_field_propagation(input_pc, patch_indices, all_patches_indices,
                                                diffuse=opts.diffuse,
                                                weights=pc_probs if opts.weighted_prop else None)

    with MyTimer('fix global orientation'):
        # STEP: fix global orientation
        if field_utils.measure_mean_potential(input_pc) < 0:
            # if average global potential is negative, flip all normals
            input_pc[:, 3:] *= -1

    MyTimer.print_total_time()
    with MyTimer('exporting result', count=False):
        util.export_pc(transform.inverse(input_pc).transpose(0, 1), opts.export_dir / f'final_result.xyz')


if __name__ == '__main__':
    opts = get_parser().parse_args()

    opts.export_dir.mkdir(exist_ok=True, parents=True)
    options.export_options(opts)
    run(opts)
