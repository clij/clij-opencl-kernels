__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void variance_box(
    IMAGE_dst_TYPE  dst,
    IMAGE_src_TYPE  src,
    const int       index0,
    const int       index1,
    const int       index2
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0)
  const int4 r = (int4){(index0-1)/2, (index1-1)/2, (index2-1)/2, 0};
  
  int count = 0;
  float sum = 0;
  for (int dx = -r.x; dx <= r.x; ++dx) {
    for (int dy = -r.y; dy <= r.y; ++dy) {
      for (int dz = -r.z; dz <= r.z; ++dz) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz,0)
          sum = sum + (float) READ_src_IMAGE(src, sampler, coord + pos).x;
          count++;
      }
    }
  }
  const float mean_intensity = sum / count;
  sum = 0;
  count = 0;
  for (int dx = -r.x; dx <= r.x; ++dx) {
    for (int dy = -r.y; dy <= r.y; ++dy) {
      for (int dz = -r.z; dz <= r.z; ++dz) {
          const POS_src_TYPE pos = POS_src_INSTANCE(dx, dy, dz,0)
          const float value = (float) READ_src_IMAGE(src, sampler, coord + pos).x;
          sum = sum + pow(value - mean_intensity, 2);
          count++;
      }
    }
  }
  WRITE_dst_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(sum / (count)));
}