using System.Numerics.Tensors;

namespace Darp.Geometry;

// Checks precede dimension indexing so rank errors are reported at the operation boundary.
internal static class MatrixShape
{
    public static void RequireMatrix(in ReadOnlyTensorSpan<double> matrix)
    {
        if (
            matrix.Rank != 2
            || matrix.Lengths[0] < 0
            || matrix.Lengths[1] < 0
            || matrix.Lengths[0] > int.MaxValue
            || matrix.Lengths[1] > int.MaxValue
        )
            throw new ArgumentException("Expected a rank-two matrix with Int32 dimensions.", nameof(matrix));
    }

    public static void RequireSameShape(in ReadOnlyTensorSpan<double> s1, in ReadOnlyTensorSpan<double> s2)
    {
        RequireMatrix(in s1);
        RequireMatrix(in s2);
        if (s1.Lengths[0] != s2.Lengths[0] || s1.Lengths[1] != s2.Lengths[1])
            throw new ArgumentException("Matrix shapes must match.", nameof(s2));
    }

    public static void RequireSameShape(in ReadOnlyTensorSpan<double> s1, in ReadOnlyTensorSpan<double> s2, in ReadOnlyTensorSpan<double> s3)
    {
        RequireMatrix(in s1);
        RequireMatrix(in s2);
        RequireMatrix(in s3);
        if (s1.Lengths[0] != s2.Lengths[0] || s1.Lengths[1] != s2.Lengths[1] || s1.Lengths[0] != s3.Lengths[0] || s1.Lengths[1] != s3.Lengths[1])
            throw new ArgumentException("Matrix shapes must match.");
    }

    public static void RequireProduct(ReadOnlyTensorSpan<double> left, ReadOnlyTensorSpan<double> right)
    {
        RequireMatrix(left);
        RequireMatrix(right);
        if (left.Lengths[1] != right.Lengths[0])
            throw new ArgumentException("Inner dimensions must match.", nameof(right));
    }

    public static void RequireVectors(ReadOnlyTensorSpan<double> left, ReadOnlyTensorSpan<double> right)
    {
        RequireSameShape(left, right);
        if (left.Lengths[1] != 1)
            throw new ArgumentException("Expected N by 1 column vectors.");
    }

    public static void RequireSize(ReadOnlyTensorSpan<double> matrix, int rows, int columns)
    {
        RequireMatrix(matrix);
        if (matrix.Lengths[0] != rows || matrix.Lengths[1] != columns)
            throw new ArgumentException($"Expected a {rows} by {columns} matrix.", nameof(matrix));
    }

    public static void RequireVector3Pair(ReadOnlyTensorSpan<double> left, ReadOnlyTensorSpan<double> right)
    {
        RequireSize(left, 3, 1);
        RequireSize(right, 3, 1);
    }
}
