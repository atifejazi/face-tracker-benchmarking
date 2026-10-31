"""Camera regression tests without model weights or CUDA."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from export_smirk_to_ga import fixed_camera_c2w, smirk_perspective_translation
from diagnose_ga_mesh_overlay import smirk_native_pixels, orientation
from ga_mesh_utils import ga_intrinsics, ga_w2c_from_transform_matrix, project_vertices


class SmirkProjectionTests(unittest.TestCase):
    def test_offcenter_crop_and_translation_match_native_on_reference_plane(self):
        # Nonzero tx/ty, off-center crop, odd/non-square image, varying scales
        # and camera depths catch both historical sign errors independently.
        for scale in (3., 8.2, 15.):
            for camera_z in (-2., 1.8, 5.):
                crop = np.array([[.7, 0, -52.], [0, .7, -37.], [0, 0, 1.]])
                cam = np.array([scale, .03, -.07])
                frame = dict(w=513, h=480, camera_angle_x=2*np.arctan(513/(2*2300)),
                             transform_matrix=fixed_camera_c2w(camera_z, y_flip=True))
                K = ga_intrinsics(frame)
                t = smirk_perspective_translation(cam, crop, K[0, 0], K[0, 2], K[1, 2], camera_z)
                verts = np.array([[-.08, .1, 0], [.06, -.1, 0], [.02, .03, 0]])
                uv, depth = project_vertices(verts+t, frame)
                np.testing.assert_allclose(uv, smirk_native_pixels(verts, cam, crop), atol=3e-5)
                self.assertTrue((depth > 0).all())
                np.testing.assert_allclose(ga_w2c_from_transform_matrix(frame['transform_matrix'])[:3, :3], np.diag([1,-1,-1]))

    def test_ga_projection_uses_actual_fov_and_rasterizer_pixel_centers(self):
        frame = dict(w=512, h=400, camera_angle_x=2*np.arctan(512/2400),
                     fl_x=9, fl_y=4, cx=3, cy=8)
        np.testing.assert_allclose(ga_intrinsics(frame), [[1200,0,255.5],[0,1200,199.5],[0,0,1]])
        np.testing.assert_allclose(ga_intrinsics(frame, (256,200)), [[600,0,127.5],[0,600,99.5],[0,0,1]])

        np.testing.assert_allclose(ga_intrinsics(frame, (256,201)), [[600,0,127.5],[0,603,100],[0,0,1]])

    def test_invalid_crop_and_camera_fail(self):
        for crop in (np.diag([-1.,-1.,1.]), np.array([[1,.1,0],[0,1,0],[0,0,1]]), np.diag([1.,2.,1.])):
            with self.assertRaises(ValueError):
                smirk_perspective_translation([8,0,0],crop,1200,256,256,1.8)
        for cam in ([0,0,0], [-1,0,0], [8,0,float('nan')]):
            with self.assertRaises(ValueError):
                smirk_perspective_translation(cam,np.eye(3),1200,256,256,1.8)

    def test_chin_brow_orientation_and_visibility(self):
        uv = np.zeros((68,2)); uv[8,1] = 100
        self.assertEqual(orientation(uv,np.ones(68)), 'UPRIGHT')
        uv[27,1] = 200
        self.assertEqual(orientation(uv,np.ones(68)), 'UPSIDE_DOWN')
        self.assertEqual(orientation(uv,-np.ones(68)), 'NOT_VISIBLE')

    def test_export_preserves_pose_and_rejects_ambiguous_old_flags(self):
        script = Path(__file__).resolve().parents[1]/'scripts/export_smirk_to_ga.py'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); canonical=root/'canonical'; canonical.mkdir()
            frame=dict(timestep_index=0, file_path='images/00000_00.png', fl_x=1200,fl_y=1200,
                       cx=254,cy=248,w=512,h=512,transform_matrix=fixed_camera_c2w(1.8).tolist())
            for split in ('train','val','test'):
                (canonical/f'transforms_{split}.json').write_text(json.dumps(dict(frames=[frame])))
            params=dict(shape=torch.zeros(1,300), exp=torch.zeros(1,50), pose=torch.tensor([[.2,-.3,.1]]),
                        jaw=torch.tensor([[.3,0.,0.]]),cam=torch.tensor([[8.,.03,.02]]),
                        image_to_crop=torch.eye(3)[None], frame_names=['00000_00'])
            track=root/'track.pt';torch.save(params,track)
            command=[sys.executable,str(script),'--track-params',str(track), '--canonical-dataset',str(canonical), '--target',str(root/'out')]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            with np.load(root/'out/flame_param/00000.npz') as data:
                np.testing.assert_array_equal(data['rotation'],params['pose'].numpy())
                np.testing.assert_array_equal(data['jaw_pose'],params['jaw'].numpy())
            metadata=json.loads((root/'out/smirk_export.json').read_text())
            self.assertEqual(metadata['projection_mode'],'smirk')
            command[-1]=str(root/'bad')
            run=subprocess.run(command+['--opencv-y-flip'],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse((root/'bad').exists())
            run=subprocess.run(command+['--projection-mode','legacy'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            with np.load(root/'bad/flame_param/00000.npz') as data:
                self.assertLess(data['translation'][0,2],0)  # legacy depth-camera_z


if __name__ == '__main__':
    unittest.main()
