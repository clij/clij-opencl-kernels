__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void binary_erosion(
    IMAGE_src_TYPE  src,
    IMAGE_src_TYPE  footprint,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const POS_src_TYPE   pos_image = POS_src_INSTANCE(    x,  y,  z, 0);
  float value = READ_IMAGE(src, sampler, pos_image).x;

  if (value == 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  const int footprint_width  = GET_IMAGE_WIDTH(footprint)  > 1 ? GET_IMAGE_WIDTH(footprint)  : 0;
  const int footprint_height = GET_IMAGE_HEIGHT(footprint) > 1 ? GET_IMAGE_HEIGHT(footprint) : 0;
  const int footprint_depth  = GET_IMAGE_DEPTH(footprint)  > 1 ? GET_IMAGE_DEPTH(footprint)  : 0;
  const int4 c = (int4){footprint_width / 2, footprint_height / 2, footprint_depth / 2, 0};
  const POS_footprint_TYPE pos_footprint = POS_footprint_INSTANCE(c.x,c.y,c.z, 0);

  for (int cz = -c.z; cz <= c.z; ++cz) {
    for (int cy = -c.y; cy <= c.y; ++cy) {
      for (int cx = -c.x; cx <= c.x; ++cx) {

        POS_footprint_TYPE coord_footprint = pos_footprint + POS_footprint_INSTANCE(cx,cy,cz,0);
        float footprint_value = (float) READ_IMAGE(footprint, sampler, coord_footprint).x;
        if (footprint_value == 0) {
          continue;
        }

        POS_src_TYPE coord_image  = pos_image  + POS_src_INSTANCE(cx,cy,cz,0);
        float image_value = (float) READ_IMAGE(src, sampler, coord_image).x;
        if (image_value == 0) {
          WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(0));
          return;
        }

      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(1));
}