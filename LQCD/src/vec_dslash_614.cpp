    const int srcO_scale = subgrid[0] * subgrid[1] * subgrid[2] * 12;
    const int destE_scale = subgrid[0] * subgrid[1] * subgrid[2] * 12;
    const int AO_scale = subgrid[0] * subgrid[1] * subgrid[2] * 9;
    const int AE_scale = subgrid[0] * subgrid[1] * subgrid[2] * 9;

    int y_u = (N_sub[1] == 1) ? subgrid[1] : subgrid[1] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < y_u; y++) {
            const int f_y = (y + 1) % subgrid[1];
            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> * const srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * f_y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> * const destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> * const AE_base = U.A[1] + (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 9;

                // index multiplied by 2: complex<double> == double[2]
                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AE_vindex = _mm_set_epi32(6 * AE_scale, 4 * AE_scale, 2 * AE_scale, 0 * AE_scale);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AE = AE_base + t * AE_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAEReal, vAEImag;
                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAEReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]), AE_vindex, gather_scale);
                            vAEImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]) + 1, AE_vindex, gather_scale);
                            vAEReal = _mm256_mul_pd(vAEReal, vNHalf); // multiply AE by -0.5: saves some instructions later
                            vAEImag = _mm256_mul_pd(vAEImag, vNHalf);
                            // vAE = -half * AE[c1 * 3 + c2]
                            // don't touch vAE from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] += flag * (tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                        }
                    }
                }

                /*
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                         subgrid[0] * f_y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
                */
            }
        }
    }

    int y_d = (N_sub[1] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = y_d; y < subgrid[1]; y++) {
            int b_y = (y - 1 + subgrid[1]) % subgrid[1];

            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> * const srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> * const destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> * const AO_base = U.A[1] +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) * 9;

                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AO_vindex = _mm_set_epi32(6 * AO_scale, 4 * AO_scale, 2 * AO_scale, 0 * AO_scale);
                const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AO = AO_base + t * AO_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAOReal, vAOImag;

                    /*
                    // PROVEN CORRECT
                    complex<double> tmp;
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (tmp);
                        }
                    }
                    */

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAOReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]), AO_vindex, gather_scale);
                            vAOImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]) + 1, AO_vindex, gather_scale);
                            vAOReal = _mm256_mul_pd(vAOReal, vNHalf);
                            vAOImag = _mm256_sub_pd(vZero, _mm256_mul_pd(vAOImag, vNHalf)); // conj(AO)
                            // vAO = conj(-half * AO[c2 * 3 + c1])
                            // don't touch vAO from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * conj(-half * AO[c2 * 3 + c1])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] -= flag * (tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[1 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (tmp);
                        }
                    }
                }
            }
        }
    }
