
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define READ_IMAGE_ZERO_OUTSIDE(a,b,c) read_buffer2duc_zero_outside(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)

inline uchar2 read_buffer2duc_zero_outside(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int2 position )
{
    int2 pos = (int2){position.x, position.y};
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (uchar2){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}

__kernel void superior_inferior(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);

  // if value is already 1, dilate will return 1
  float value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, pos).x;
  // if (value != 0) {
  //   WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
  //   return;
  // }

  /* Dilate with kernel [[1, 0, 0], 
                         [0, 1, 0], 
                         [0, 0, 1]] */
  value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(1, 1, 0, 0))).x;
  if (value == 0) {
    value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(-1, -1, 0, 0))).x;
    if (value == 0) {
      WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
      return;
    }
  }

  /* Dilate with kernel [[0, 1, 0], 
                         [0, 1, 0], 
                         [0, 1, 0]] */
  value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(0, 1, 0, 0))).x;
    if (value == 0) {
      value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(0, -1, 0, 0))).x;
      if (value == 0) {
        WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  /* Dilate with kernel [[0, 0, 1], 
                         [0, 1, 0], 
                         [1, 0, 0]] */
  value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(-1, 1, 0, 0))).x;
    if (value == 0) {
      value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(1, -1, 0, 0))).x;
      if (value == 0) {
        WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  /* Dilate with kernel [[0, 0, 0], 
                         [1, 1, 1], 
                         [0, 0, 0]] */
  value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(1, 0, 0, 0))).x;
    if (value == 0) {
      value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(-1, 0, 0, 0))).x;
      if (value == 0) {
        WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
        return;
      }
    }

  // If all dilates are 1 then return 1
  WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
}
