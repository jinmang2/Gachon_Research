# Gachon Research
This repo describes a python implementation of three methodologies:
  - Sigma & Gamma Convergence
  - Experience Curve
  - Scaling
# Do Projects
- Globalization
- Particulate Matter 2.5
- Sanitation
- Marine Protected Area
- Suicide
- Panel Regression on PM2.5
- PACOV
- Metropolitan PM2.5
- Corruption

# About Methodology
## Sigma & Gamma Convergence
&nbsp;&nbsp;&nbsp;&nbsp;The convergence analysis attempts to examine two basis questions. First, do countries
initially lagging in such performance measures as globalization index tend to improve faster so
that they catch up to the performance of leading countries over time? Second, does dispersion of
globalization index among countries get reduced over time?<br>

&nbsp;&nbsp;&nbsp;&nbsp;Traditionally, β convergence is used to examine the first question, while σ convergence is
used to analyze the second question. The so-called Barro β convergence method (Barro, 1991)
regress the rate of improvement during a period on the initial value of the performance measure
for respective countries. If the value of coefficient of slope is negative and statistically significant,
then the catch-up process is demonstrated (Barro, 1991; Barro and Sala-i-Martin, 1992).<br>

&nbsp;&nbsp;&nbsp;&nbsp;For our methodology, we use γ convergence (Boyle and McCarthy, 1997) and σ
convergence (Friedman, 1992). Common measures of dispersion include standard deviation and
coefficient of variation (Heckelman, 2015). For σ convergence, we have selected to use coefficient
of variation (CV). CV is measured by dividing standard deviation by the sample average. Using CV
which is dimensionless ratio enables us to compare the degree of dispersion for performance
measures with different units. We then measure the inter-temporal changes by normalizing CV in
subsequent years to CV at the initial year of 1991. Therefore, CV in 1991 is always 1.0. If CV in
subsequent years is less than CV in the initial year, then, the normalized CV in subsequent years
will be less than 1.0. If the values of normalized CVs in the subsequent years continue to decrease,
and the differences between CVs are statistically significant, the result is viewed as evidence of σ
convergence or reduction of dispersion. We use two sample t tests for CV (http://www.realstatistics.com/students-t-distribution/coefficient-of-variation-testing/). This test works best when
the sample sizes are at least 10. Since our sample sizes are much larger than 10, this test should
work well.<br>

&nbsp;&nbsp;&nbsp;&nbsp;For γ convergence model, Boyle and McCarthy (1997) suggested the use of Kendall’s
index of rank concordance which measures mobility of the individual countries over time within
the cross country distribution of a particular performance measure (Liddle, 2012; Chang et al.,
forthcoming). In other words, γ convergence measures the degree of changing ranking order of
countries between a given year and the initial year. The γ-convergence we use is Kendall’s binary
index version and is defined as follows:<br>

![image](https://github.com/jinmang2/Gachon_Research/blob/master/img/gamma.PNG?raw=true)

<br>&nbsp;&nbsp;Where 𝐴𝑅 (𝑌) = the actual rank of country i’s performance measure in year t
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;𝐴𝑅 (0) = the actual rank of country i’s performance measure in year 0
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;γ_𝑡 = Binary Gamma Index in year t.<br>

&nbsp;&nbsp;&nbsp;&nbsp;The γ index has the advantage of being of single number traced over time in twodimension, analogous to the σ convergence index. The value of rank concordance ranges from
zero to unity. If no change in rank order takes place, the rank concordance becomes unity. If a
catch-up process is present, which result in change of rank order the index will be less than unity.
The statistic is distributed as chi-square and we test the null hypothesis that γ convergence shows
no difference between ranks of different years (Siegel, 1956).<br>

&nbsp;&nbsp;&nbsp;&nbsp;According to Real Statistics Using Excel (http://www.real-statistics.com/reliability/kendallsw/), the proper use of X2
test to test statistical difference between Kendall’s coefficients of
concordance (W) on yearly γ indexes requires that the number of countries involved should be
equal to 5 or more. Or the number of years being compared should be more than 15 years. In
our case, the number of countries involved will be much larger than 5 countries. Therefore, we
can use this X2
test to validate the null hypothesis that W=0 or that there is no agreement
between the years being compared.<br>

&nbsp;&nbsp;&nbsp;&nbsp;How do we use σ and γ index together to evaluate reduction of dispersion as well as
catch-up process? There are four different cases that can occur. The simplest case is when both σ
and γ index are increasing in values. Under the circumstance, neither reduction of dispersion nor
catch-up may be taking place. The second case is that both σ and γ indexes are decreasing which
indicates that both reduction of dispersion and catch-up process are taking place. The third case
occurs where σ convergence measure is non-decreasing, while γ convergence value is in decline.
Since β convergence is a necessary but not sufficient condition for σ convergence, this indicates
that catch-up process is taking place, while reduction of dispersion is not. The fourth case occurs
where γ index is non-decreasing, while a substantial decline occurred with σ index. This indicates
that country differences in performance measures remain so that no rank change among
countries takes place. However, performance differences among countries have reduced
considerably, which indicates conditional β convergence. Put it another way, catch-up process may
be taking place within subgroups of countries. 

## Experience Curve
- reference: Kinked Experience Curve, YuSang Chang
![title](https://github.com/jinmang2/Gachon_Research/blob/master/img/Figure%208.%20Kinked%20Experience%20Curves%20for%20Energy%20Intensity%20of%20US%20and%20Japan.PNG?raw=true)


## Scaling Methodology
- Panel Regression, Region Analysis, PCSE, GLS, random/fixed effect
