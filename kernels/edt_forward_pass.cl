__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void edt_forward_pass_x(
    IMAGE_src_TYPE  src
)
{
  // const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  for (int x = 0; x < width; x++) {

    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
    IMAGE_src_PIXEL_TYPE temp = 0;
    if (value == 0) {
      continue;
    }
    if(x > 0)
    {
      value = READ_IMAGE(src, sampler, pos).x;
      temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y, z, 0)).x + 1;
      WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    }
    // if(y > 0)
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y - 1, z, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }
    // if(z > 0)
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z - 1, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }
    // if (x > 0 && y > 0) 
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y - 1, z, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }
    // if (x > 0 && z > 0) 
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y, z - 1, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }
    // if (y > 0 && z > 0) 
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y - 1, z - 1, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }
    // if (x > 0 && y > 0 && z > 0) 
    // {
    //   value = READ_IMAGE(src, sampler, pos).x;
    //   temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y - 1, z - 1, 0)).x + 1;
    //   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(min(value, temp)));
    // }

  }
}