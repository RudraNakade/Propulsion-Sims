import numpy as np

lc = 189.70

rc = 92 / 2
rt = 42.41 / 2

rcyl = 50
rconv = 1.5 * rt
rdiv = 0.382 * rt

theta_conv = np.deg2rad(30.0)

print(f'''Chamber length: {lc:.2f} mm
      Chamber radius: {rc:.2f} mm
      Throat radius: {rt:.2f} mm
      Cylindrical radius: {rcyl:.2f} mm
      Converging radius: {rconv:.2f} mm
      Diverging radius: {rdiv:.2f} mm''')

l_cyl_arc = rcyl * np.sin(theta_conv)
l_conv_arc = rconv * np.sin(theta_conv)
print(f"Cylindrical arc length: {l_cyl_arc:.4f} mm")
print(f"Converging arc length: {l_conv_arc:.4f} mm")

dr_cyl_arc = rcyl * (1 - np.cos(theta_conv))
dr_conv_arc = rconv * (1 - np.cos(theta_conv))
print(f"delta r - cylindrical: {dr_cyl_arc:.4f} mm")
print(f"delta r - converging: {dr_conv_arc:.4f} mm")

dr_cone = rc - rt - dr_cyl_arc - dr_conv_arc
dl_cone = dr_cone / np.tan(theta_conv)
print(f"delta r - cone: {dr_cone:.4f} mm")
print(f"delta l - cone: {dl_cone:.4f} mm")