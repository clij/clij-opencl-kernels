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

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Assuming height <= 256

  // Load data into local memory
  for (int x = 0; y < width; x++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    local_mem[x] = READ_IMAGE(src, sampler, pos).x;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Forward pass
  for (int x = 0; x < width; x++) {
    IMAGE_src_PIXEL_TYPE value = local_mem[y];
    if (value != 0 && y > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[x - 1] + 1;
      value = min(value, temp);
    }
    local_mem[x] = value;
  }

  // Backward pass
  for (int x = width - 1; x >= 0; x--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[y];
    if (value != 0 && y < (height - 1)) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[x + 1] + 1;
      value = min(value, temp);
    }
    local_mem[x] = value;
  }

  // Write data back to global memory
  for (int x = 0; x < height; x++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(local_mem[x]));
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

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Assuming height <= 256

  // Load data into local memory
  for (int y = 0; y < height; y++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    local_mem[y] = READ_IMAGE(src, sampler, pos).x;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Forward pass
  for (int y = 0; y < height; y++) {
    IMAGE_src_PIXEL_TYPE value = local_mem[y];
    if (value != 0 && y > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[y - 1] + 1;
      value = min(value, temp);
    }
    local_mem[y] = value;
  }

  // Backward pass
  for (int y = height - 1; y >= 0; y--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[y];
    if (value != 0 && y < (height - 1)) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[y + 1] + 1;
      value = min(value, temp);
    }
    local_mem[y] = value;
  }

  // Write data back to global memory
  for (int y = 0; y < height; y++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(local_mem[y]));
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

  __local IMAGE_src_PIXEL_TYPE local_mem[LOCAL_MEM_SIZE]; // Assuming depth <= 256

  // Load data into local memory
  for (int z = 0; z < depth; z++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    local_mem[z] = READ_IMAGE(src, sampler, pos).x;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Forward pass
  for (int z = 0; z < depth; z++) {
    IMAGE_src_PIXEL_TYPE value = local_mem[z];
    if (value != 0 && z > 0) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[z - 1] + 1;
      value = min(value, temp);
    }
    local_mem[z] = value;
  }

  // Backward pass
  for (int z = depth - 1; z >= 0; z--) {
    IMAGE_src_PIXEL_TYPE value = local_mem[z];
    if (value != 0 && z < (depth - 1)) {
      IMAGE_src_PIXEL_TYPE temp = local_mem[z + 1] + 1;
      value = min(value, temp);
    }
    local_mem[z] = value;
  }

  // Write data back to global memory
  for (int z = 0; z < depth; z++) {
    const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
    WRITE_IMAGE(src, pos, CONVERT_src_PIXEL_TYPE(local_mem[z]));
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