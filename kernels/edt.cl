__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void edt_pass_x(
    IMAGE_src_TYPE  src
)
{
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Adjust size as needed

  // First pass: left to right
  for (int x = 0; x <= width; x++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;

    if (value != 0 && x > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[x - 1] + 1;
      value = min(value, temp);
    }

    local_mem[x] = value;
  }

  // Second pass: right to left
  for (int x = width; x >= 0; x--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[x];

    if (value != 0 && x < width) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[x + 1] + 1;
      value = min(value, temp);
    }

    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
  }
}


__kernel void edt_pass_y(
    IMAGE_src_TYPE  src
)
{
  const int x = get_global_id(0);
  const int z = get_global_id(2);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Adjust size as needed

  // First pass: left to right
  for (int y = 0; y <= height; y++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;

    if (value != 0 && y > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[y - 1] + 1;
      value = min(value, temp);
    }

    local_mem[y] = value;
  }

  // Second pass: right to left
  for (int y = height; y >= 0; y--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[y];

    if (value != 0 && y < height) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[y + 1] + 1;
      value = min(value, temp);
    }

    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
  }
}

__kernel void edt_pass_z(
    IMAGE_src_TYPE  src
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);
  const int depth = GET_IMAGE_DEPTH(src);

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Adjust size as needed

  // First pass: left to right
  for (int z = 0; z <= depth; z++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;

    if (value != 0 && z > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[z - 1] + 1;
      value = min(value, temp);
    }

    local_mem[z] = value;
  }

  // Second pass: right to left
  for (int z = depth; z >= 0; z--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[z];

    if (value != 0 && z < depth) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[z + 1] + 1;
      value = min(value, temp);
    }

    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
  }
}


// __kernel void edt_pass_x(
//     IMAGE_src_TYPE  src
// )
// {
//   const int y = get_global_id(1);
//   const int z = get_global_id(2);

//   const int width = GET_IMAGE_WIDTH(src);
//   const int height = GET_IMAGE_HEIGHT(src);
//   const int depth = GET_IMAGE_DEPTH(src);

//   for (int x = 0; x <= width; x++) {
//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;

//     if (value == 0) continue;

//     if(x > 0)
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x - 1, y, z, 0)).x + 1;
//       value = min(value, temp);
//     }

//     WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }

//   for (int x = width; x >= 0 ; x--) {
//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;

//     if (value == 0) continue;

//     if(x < (width - 1))
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x + 1, y, z, 0)).x + 1;
//       value = min(value, temp);
//     }

//   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }
// }

// __kernel void edt_pass_y(
//     IMAGE_src_TYPE  src
// )
// {
//   const int x = get_global_id(0);
//   const int z = get_global_id(2);

//   const int width = GET_IMAGE_WIDTH(src);
//   const int height = GET_IMAGE_HEIGHT(src);
//   const int depth = GET_IMAGE_DEPTH(src);

//   for (int y = 0; y <= height; y++) {

//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
//     if (value == 0) continue;

//     if(y > 0)
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y - 1, z, 0)).x + 1;
//       value = min(value, temp);
//     }

//     WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }

//   for (int y = height; y >= 0; y--) {

//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
//     if (value == 0) continue;

//     if(y < (height - 1))
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y + 1, z, 0)).x + 1;
//       value = min(value, temp);
//     }

//     WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }
// }

// __kernel void edt_pass_z(
//     IMAGE_src_TYPE  src
// )
// {
//   const int x = get_global_id(0);
//   const int y = get_global_id(1);

//   const int width = GET_IMAGE_WIDTH(src);
//   const int height = GET_IMAGE_HEIGHT(src);
//   const int depth = GET_IMAGE_DEPTH(src);

//   for (int z = 0; z <= depth; z++) {

//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
//     if (value == 0) continue;

//     if(z > 0)
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z - 1, 0)).x + 1;
//       value = min(value, temp);
//     }

//   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }

//   for (int z = depth; z >= 0; z--) {

//     const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
//     IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, pos).x;
//     if (value == 0) continue;

//     if(z < (depth - 1))
//     {
//       IMAGE_src_PIXEL_TYPE temp = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z + 1, 0)).x + 1;
//       value = min(value, temp);
//     }

//   WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(value));
//   }
// }