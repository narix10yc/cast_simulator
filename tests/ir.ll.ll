; ModuleID = '../tests/ir.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s1_sep_u2q_k2l0_ffffffffffffffff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %left = shl i64 %idx2, 1
  %idxHi = or i64 %left, 1
  %pReLo = getelementptr <4 x double>, ptr %preal, i64 %left
  %pReHi = getelementptr <4 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <4 x double>, ptr %pimag, i64 %left
  %pImHi = getelementptr <4 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <4 x double>, ptr %pReLo, align 32
  %ReHi = load <4 x double>, ptr %pReHi, align 32
  %ImLo = load <4 x double>, ptr %pImLo, align 32
  %ImHi = load <4 x double>, ptr %pImHi, align 32
  %Re0 = shufflevector <4 x double> %ReLo, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Re1 = shufflevector <4 x double> %ReLo, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Re2 = shufflevector <4 x double> %ReHi, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Re3 = shufflevector <4 x double> %ReHi, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Im0 = shufflevector <4 x double> %ImLo, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Im1 = shufflevector <4 x double> %ImLo, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Im2 = shufflevector <4 x double> %ImHi, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Im3 = shufflevector <4 x double> %ImHi, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  %newReLo = shufflevector <2 x double> %newRe0, <2 x double> %newRe1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newReHi = shufflevector <2 x double> %newRe2, <2 x double> %newRe3, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newReLo, ptr %pReLo, align 32
  store <4 x double> %newReHi, ptr %pReHi, align 32
  %newImLo = shufflevector <2 x double> %newIm0_35, <2 x double> %newIm1_42, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newImHi = shufflevector <2 x double> %newIm2_49, <2 x double> %newIm3_56, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newImLo, ptr %pImLo, align 32
  store <4 x double> %newImHi, ptr %pImHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u2q_k1l0_ffffffffffffffff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx2
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx2
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <4 x double> %mRe4, %Re0
  %newRe1_8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = fmul <4 x double> %mIm4, %Im0
  %newRe1_12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <4 x double> %mRe8, %Re0
  %newRe2_15 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = fmul <4 x double> %mIm8, %Im0
  %newRe2_19 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <4 x double> %mRe12, %Re0
  %newRe3_22 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = fmul <4 x double> %mIm12, %Im0
  %newRe3_26 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  %vecRe0 = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_35, <4 x double> %newIm1_42, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_49, <4 x double> %newIm3_56, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u2q_k5l3_ffffffffffffffff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx4 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %left_tmp = shl i64 %idx4, 4
  %left = and i64 %left_tmp, -64
  %middle_tmp = shl i64 %idx4, 3
  %middle = and i64 %middle_tmp, 16
  %right_tmp = shl i64 %idx4, 2
  %right = and i64 %right_tmp, 4
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <4 x double> %mRe4, %Re0
  %newRe1_8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = fmul <4 x double> %mIm4, %Im0
  %newRe1_12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <4 x double> %mRe8, %Re0
  %newRe2_15 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = fmul <4 x double> %mIm8, %Im0
  %newRe2_19 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <4 x double> %mRe12, %Re0
  %newRe3_22 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = fmul <4 x double> %mIm12, %Im0
  %newRe3_26 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_35, ptr %pIm0, align 32
  store <4 x double> %newIm1_42, ptr %pIm1, align 32
  store <4 x double> %newIm2_49, ptr %pIm2, align 32
  store <4 x double> %newIm3_56, ptr %pIm3, align 32
  %idx_next = add nsw i64 %idx4, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s1_sep_u2q_k2l1_ffffffffffffffff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx4 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %left = shl i64 %idx4, 3
  %idx1 = or i64 %left, 2
  %idx2 = or i64 %left, 4
  %idx3 = or i64 %left, 6
  %pRe0 = getelementptr double, ptr %preal, i64 %left
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %left
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <2 x double>, ptr %pRe0, align 16
  %Re1 = load <2 x double>, ptr %pRe1, align 16
  %Re2 = load <2 x double>, ptr %pRe2, align 16
  %Re3 = load <2 x double>, ptr %pRe3, align 16
  %Im0 = load <2 x double>, ptr %pIm0, align 16
  %Im1 = load <2 x double>, ptr %pIm1, align 16
  %Im2 = load <2 x double>, ptr %pIm2, align 16
  %Im3 = load <2 x double>, ptr %pIm3, align 16
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  store <2 x double> %newRe0, ptr %pRe0, align 16
  store <2 x double> %newRe1, ptr %pRe1, align 16
  store <2 x double> %newRe2, ptr %pRe2, align 16
  store <2 x double> %newRe3, ptr %pRe3, align 16
  store <2 x double> %newIm0_35, ptr %pIm0, align 16
  store <2 x double> %newIm1_42, ptr %pIm1, align 16
  store <2 x double> %newIm2_49, ptr %pIm2, align 16
  store <2 x double> %newIm3_56, ptr %pIm3, align 16
  %idx_next = add nsw i64 %idx4, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #1

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_alt_u3_k3_33330333(ptr nocapture %psv, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %bi_n = fmul <4 x double> %bi, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %ci_n = fmul <4 x double> %ci, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %di_n = fmul <4 x double> %di, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %and_outer = shl i64 %idx2, 3
  %shl_outer = and i64 %and_outer, -32
  %and_inner = shl i64 %idx2, 2
  %shl_inner = and i64 %and_inner, 12
  %idxA = or i64 %shl_outer, %shl_inner
  %idxB = or i64 %idxA, 16
  %ptrLo = getelementptr double, ptr %psv, i64 %idxA
  %ptrHi = getelementptr double, ptr %psv, i64 %idxB
  %Lo = load <4 x double>, ptr %ptrLo, align 32
  %Hi = load <4 x double>, ptr %ptrHi, align 32
  %LoRe = fmul <4 x double> %ar, %Lo
  %LoRe1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Hi, <4 x double> %LoRe)
  %LoIm_s = fmul <4 x double> %bi_n, %Hi
  %LoIm = shufflevector <4 x double> %LoIm_s, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newLo = fadd <4 x double> %LoRe1, %LoIm
  %HiRe = fmul <4 x double> %cr, %Lo
  %HiRe2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Hi, <4 x double> %HiRe)
  %HiIm_s = fmul <4 x double> %ci_n, %Lo
  %HiIm_s3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_n, <4 x double> %Hi, <4 x double> %HiIm_s)
  %HiIm = shufflevector <4 x double> %HiIm_s3, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newHi = fadd <4 x double> %HiRe2, %HiIm
  store <4 x double> %newLo, ptr %ptrLo, align 32
  store <4 x double> %newHi, ptr %ptrHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k3_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %and_outer = shl i64 %idx2, 3
  %shl_outer = and i64 %and_outer, -16
  %and_inner = shl i64 %idx2, 2
  %shl_inner = and i64 %and_inner, 4
  %idxA = or i64 %shl_outer, %shl_inner
  %idxB = or i64 %idxA, 8
  %pAr = getelementptr double, ptr %preal, i64 %idxA
  %pAi = getelementptr double, ptr %pimag, i64 %idxA
  %pBr = getelementptr double, ptr %preal, i64 %idxB
  %pBi = getelementptr double, ptr %pimag, i64 %idxB
  %Ar = load <4 x double>, ptr %pAr, align 32
  %Ai = load <4 x double>, ptr %pAi, align 32
  %Br = load <4 x double>, ptr %pBr, align 32
  %Bi = load <4 x double>, ptr %pBi, align 32
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  store <4 x double> %newAr2, ptr %pAr, align 32
  store <4 x double> %newAi4, ptr %pAi, align 32
  store <4 x double> %newBr7, ptr %pBr, align 32
  store <4 x double> %newBi10, ptr %pBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k0_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %pRe = getelementptr <8 x double>, ptr %preal, i64 %idx2
  %pIm = getelementptr <8 x double>, ptr %pimag, i64 %idx2
  %Re = load <8 x double>, ptr %pRe, align 64
  %Im = load <8 x double>, ptr %pIm, align 64
  %Ar = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Ai = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Br = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Bi = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  %newRe = shufflevector <4 x double> %newAr2, <4 x double> %newBr7, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newIm = shufflevector <4 x double> %newAi4, <4 x double> %newBi10, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newRe, ptr %pRe, align 64
  store <8 x double> %newIm, ptr %pIm, align 64
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k1_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %pRe = getelementptr <8 x double>, ptr %preal, i64 %idx2
  %pIm = getelementptr <8 x double>, ptr %pimag, i64 %idx2
  %Re = load <8 x double>, ptr %pRe, align 64
  %Im = load <8 x double>, ptr %pIm, align 64
  %Ar = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Ai = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Br = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Bi = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  %newRe = shufflevector <4 x double> %newAr2, <4 x double> %newBr7, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newIm = shufflevector <4 x double> %newAi4, <4 x double> %newBi10, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newRe, ptr %pRe, align 64
  store <8 x double> %newIm, ptr %pIm, align 64
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_alt_u3_k1_33330333(ptr nocapture %psv, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %bi_n = fmul <4 x double> %bi, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %ci_n = fmul <4 x double> %ci, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %di_n = fmul <4 x double> %di, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %shl_outer = shl i64 %idx2, 3
  %idxB = or i64 %shl_outer, 4
  %ptrLo = getelementptr double, ptr %psv, i64 %shl_outer
  %ptrHi = getelementptr double, ptr %psv, i64 %idxB
  %Lo = load <4 x double>, ptr %ptrLo, align 32
  %Hi = load <4 x double>, ptr %ptrHi, align 32
  %LoRe = fmul <4 x double> %ar, %Lo
  %LoRe1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Hi, <4 x double> %LoRe)
  %LoIm_s = fmul <4 x double> %bi_n, %Hi
  %LoIm = shufflevector <4 x double> %LoIm_s, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newLo = fadd <4 x double> %LoRe1, %LoIm
  %HiRe = fmul <4 x double> %cr, %Lo
  %HiRe2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Hi, <4 x double> %HiRe)
  %HiIm_s = fmul <4 x double> %ci_n, %Lo
  %HiIm_s3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_n, <4 x double> %Hi, <4 x double> %HiIm_s)
  %HiIm = shufflevector <4 x double> %HiIm_s3, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newHi = fadd <4 x double> %HiRe2, %HiIm
  store <4 x double> %newLo, ptr %ptrLo, align 32
  store <4 x double> %newHi, ptr %ptrHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_alt_u3_k0_33330333(ptr nocapture %psv, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %bi_n = fmul <4 x double> %bi, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %ci_n = fmul <4 x double> %ci, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %di_n = fmul <4 x double> %di, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %shl_inner = shl i64 %idx2, 2
  %idxB = or i64 %shl_inner, 2
  %ptrLo = getelementptr double, ptr %psv, i64 %shl_inner
  %ptrHi = getelementptr double, ptr %psv, i64 %idxB
  %Lo = load <4 x double>, ptr %ptrLo, align 32
  %Hi = load <4 x double>, ptr %ptrHi, align 32
  %LoRe = fmul <4 x double> %ar, %Lo
  %LoRe1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Hi, <4 x double> %LoRe)
  %LoIm_s = fmul <4 x double> %bi_n, %Hi
  %LoIm = shufflevector <4 x double> %LoIm_s, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newLo = fadd <4 x double> %LoRe1, %LoIm
  %HiRe = fmul <4 x double> %cr, %Lo
  %HiRe2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Hi, <4 x double> %HiRe)
  %HiIm_s = fmul <4 x double> %ci_n, %Lo
  %HiIm_s3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_n, <4 x double> %Hi, <4 x double> %HiIm_s)
  %HiIm = shufflevector <4 x double> %HiIm_s3, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newHi = fadd <4 x double> %HiRe2, %HiIm
  store <4 x double> %newLo, ptr %ptrLo, align 32
  store <4 x double> %newHi, ptr %ptrHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #1

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
