
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define READ_IMAGE_ZERO_OUTSIDE(a,b,c) read_buffer3duc_zero_outside(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)

inline uchar2 read_buffer3duc_zero_outside(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int4 position )
{
    int4 pos = POS_src_INSTANCE(position.x, position.y, position.), 0};
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (uchar2){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}


__kernel void inferior_superior(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);

  // if value is already 0, erode will return 0
  float value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, pos).x;
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
    return;
  }

  // P0
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, j, 0, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P1
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, 0, j, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P2
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(0, i, j, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P3
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, j, j, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P4
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(j, i, -i, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P5
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, j, i, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P6
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, j, -i, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P7
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, i, j, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // P8
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        value = READ_IMAGE_ZERO_OUTSIDE(src, sampler, (pos + POS_src_INSTANCE(i, -i, j, 0))).x;
        if (value == 0) {
          break;
        }
      }
      if (value == 0) {
        break;
      }
    }
  if (value != 0) {
    WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(1));
    return;
  }

  // If all erodes are 0 then return 0
  WRITE_IMAGE(dst, pos, CONVERT_dst_PIXEL_TYPE(0));
}