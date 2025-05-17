package moa.classifiers.meta.DEMS;

import moa.AbstractMOAObject;
import moa.core.Utils;

public class SortingInformation extends AbstractMOAObject {

    private double classifierAcc;

    private double[] votes;

    private double nodeWeight;

    public double getNodeWeight() {
        return nodeWeight;
    }

    public void setNodeWeight(double nodeWeight) {
        this.nodeWeight = nodeWeight;
    }

    public double getNodeDepth() {
        return nodeDepth;
    }

    public void setNodeDepth(double nodeDepth) {
        this.nodeDepth = nodeDepth;
    }

    public double getTreeSize() {
        return treeSize;
    }

    public void setTreeSize(double treeSize) {
        this.treeSize = treeSize;
    }

    private double nodeDepth;
    private double treeSize;

    private double sortingValue;

    private int classifierIndex;

    public double getClassifierAcc() {
        return classifierAcc;
    }

    public void setClassifierAcc(double classifierAcc) {
        this.classifierAcc = classifierAcc;
    }

    public SortingInformation() {
        setNodeWeight(1);
        setNodeDepth(1);
        setTreeAcc(1);
        setVotes(new double[1]);
        setTreeSize(1);
        setClassifierAcc(1);
    }

    public SortingInformation(double classifierAcc, double[] votes) {
        this.classifierAcc = classifierAcc;
        this.votes = votes;
    }


    public SortingInformation(double classifierAcc, double[] votes, int classifierIndex) {
        this.classifierAcc = classifierAcc;
        this.votes = votes;
        this.classifierIndex = classifierIndex;
    }

    public double getTreeAcc() {
        return classifierAcc;
    }

    public void setTreeAcc(double treeAcc) {
        this.classifierAcc = treeAcc;
    }

    public double[] getVotes() {
        return votes;
    }

    public void setVotes(double[] votes) {
        this.votes = votes;
    }

    public double getConfidence() {
        return Utils.sum(this.votes) == 0 ? 0 : this.votes[Utils.maxIndex(this.votes)] / Utils.sum(this.votes);
    }

    public int getClassifierIndex() {
        return classifierIndex;
    }

    public void setClassifierIndex(int classifierIndex) {
        this.classifierIndex = classifierIndex;
    }

    public double getMargin() {
        if (this.votes == null || Utils.sum(this.votes) == 0) return 0;
        double max = Double.MIN_VALUE;
        double secondMax = Double.MIN_VALUE;
        if (this.votes.length == 0) {
            return 0;
        } else if (this.votes.length == 1) {
            max = this.votes[0];
            secondMax = 0;
        } else if (this.votes.length == 2) {
            max = this.votes[Utils.maxIndex(this.votes)];
            secondMax = Utils.sum(this.votes) - max;
        } else {
            for (double v : this.votes) {
                if (v > max) {
                    secondMax = max;
                    max = v;
                } else if (v <= max && v > secondMax) {
                    secondMax = v;
                }
            }
        }

        return (max - secondMax) / Utils.sum(this.votes);
    }


    public double getConfidence_TreeAcc(){
        return getClassifierAcc() * getConfidence();
    }

    public double getMargin_TreeAcc(){
        return getClassifierAcc() * getMargin();
    }

    public void setSortingValue(double value) {this.sortingValue = value;}

    public double getSortingValue() {return this.sortingValue;}

    @Override
    public void getDescription(StringBuilder sb, int indent) {

    }
}
