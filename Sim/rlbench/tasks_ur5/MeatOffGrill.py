  - 'chicken': Before alignment, move to above_grasp_pos (grasp_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<chicken's quaternion>, approach_direction="down" (default: "down"), yaw_mode="parallel".
  - 'grill': No alignment needed. Use ur5_grasp_at directly with grasp_pos=<grill's position> and approach={'axis':'z', 'distance':0.15}.


  - 'steak': Before alignment, move to above_place_pos (place_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<steak's quaternion>, approach_direction="up" (default: "down"), yaw_mode="parallel".
  - 'chicken': Before alignment, move to above_place_pos (place_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<chicken's quaternion>, approach_direction="up" (default: "down"), yaw_mode="parallel".
  - 'grill': No alignment needed. Use ur5_release_at directly with place_pos=<grill's position> and approach={'axis':'z', 'distance':0.15}.


  - 'steak': Before alignment, move to above_push_pos (push_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<steak's quaternion>, approach_direction="down" (default: "down"), yaw_mode="parallel".
  - 'chicken': Before alignment, move to above_push_pos (push_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<chicken's quaternion>, approach_direction="down" (default: "down"), yaw_mode="parallel".
  - 'grill': No alignment needed. Use ur5_push_at directly with push_pos=<grill's position> and approach={'axis':'z', 'distance':0.15}.


  - 'steak': Before alignment, move to above_press_pos (press_pos + [0,0,0.15]). Use ur5_align_gripper with reference_quat=<steak's quaternion>, approach_direction="down" (default: "down"), yaw