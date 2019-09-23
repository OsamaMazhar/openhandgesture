// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_ONESKELETON_OP_SKELETON_H_
#define FLATBUFFERS_GENERATED_ONESKELETON_OP_SKELETON_H_

#include "flatbuffers/flatbuffers.h"

namespace op_skeleton {

struct Point;

struct one_skeleton;

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) Point FLATBUFFERS_FINAL_CLASS {
 private:
  int32_t x_;
  int32_t y_;
  int32_t z_;

 public:
  Point() {
    memset(this, 0, sizeof(Point));
  }
  Point(int32_t _x, int32_t _y, int32_t _z)
      : x_(flatbuffers::EndianScalar(_x)),
        y_(flatbuffers::EndianScalar(_y)),
        z_(flatbuffers::EndianScalar(_z)) {
  }
  int32_t x() const {
    return flatbuffers::EndianScalar(x_);
  }
  int32_t y() const {
    return flatbuffers::EndianScalar(y_);
  }
  int32_t z() const {
    return flatbuffers::EndianScalar(z_);
  }
};
FLATBUFFERS_STRUCT_END(Point, 12);

struct one_skeleton FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_NAME = 4,
    VT_COORDINATES = 6,
    VT_NUMBER_OF_POINTS = 8
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::Vector<const Point *> *coordinates() const {
    return GetPointer<const flatbuffers::Vector<const Point *> *>(VT_COORDINATES);
  }
  int32_t number_of_points() const {
    return GetField<int32_t>(VT_NUMBER_OF_POINTS, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_COORDINATES) &&
           verifier.VerifyVector(coordinates()) &&
           VerifyField<int32_t>(verifier, VT_NUMBER_OF_POINTS) &&
           verifier.EndTable();
  }
};

struct one_skeletonBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(one_skeleton::VT_NAME, name);
  }
  void add_coordinates(flatbuffers::Offset<flatbuffers::Vector<const Point *>> coordinates) {
    fbb_.AddOffset(one_skeleton::VT_COORDINATES, coordinates);
  }
  void add_number_of_points(int32_t number_of_points) {
    fbb_.AddElement<int32_t>(one_skeleton::VT_NUMBER_OF_POINTS, number_of_points, 0);
  }
  explicit one_skeletonBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  one_skeletonBuilder &operator=(const one_skeletonBuilder &);
  flatbuffers::Offset<one_skeleton> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<one_skeleton>(end);
    return o;
  }
};

inline flatbuffers::Offset<one_skeleton> Createone_skeleton(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::Vector<const Point *>> coordinates = 0,
    int32_t number_of_points = 0) {
  one_skeletonBuilder builder_(_fbb);
  builder_.add_number_of_points(number_of_points);
  builder_.add_coordinates(coordinates);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<one_skeleton> Createone_skeletonDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const std::vector<Point> *coordinates = nullptr,
    int32_t number_of_points = 0) {
  return op_skeleton::Createone_skeleton(
      _fbb,
      name ? _fbb.CreateString(name) : 0,
      coordinates ? _fbb.CreateVectorOfStructs<Point>(*coordinates) : 0,
      number_of_points);
}

inline const op_skeleton::one_skeleton *Getone_skeleton(const void *buf) {
  return flatbuffers::GetRoot<op_skeleton::one_skeleton>(buf);
}

inline const op_skeleton::one_skeleton *GetSizePrefixedone_skeleton(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<op_skeleton::one_skeleton>(buf);
}

inline bool Verifyone_skeletonBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<op_skeleton::one_skeleton>(nullptr);
}

inline bool VerifySizePrefixedone_skeletonBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<op_skeleton::one_skeleton>(nullptr);
}

inline void Finishone_skeletonBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<op_skeleton::one_skeleton> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedone_skeletonBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<op_skeleton::one_skeleton> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace op_skeleton

#endif  // FLATBUFFERS_GENERATED_ONESKELETON_OP_SKELETON_H_
