__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void write_values_to_positions(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int index = get_global_id(0);

  const int x = (int) READ_IMAGE(src, sampler, POS_src_INSTANCE(index,0,0,0)).x;
  const int y = (int) READ_IMAGE(src, sampler, POS_src_INSTANCE(index,1,0,0)).x;
  const int z = (int) READ_IMAGE(src, sampler, POS_src_INSTANCE(index,2,0,0)).x;
  const IMAGE_src_PIXEL_TYPE value= READ_IMAGE(src, sampler, POS_src_INSTANCE(index,3,0,0)).x;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}